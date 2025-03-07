import numpy as np
import os
import shutil
from stt import logger


_silero_vad_model = {}
_has_onnx = None


def remove_non_speech(
    audio,
    use_sample=False,
    min_speech_duration=0.1,
    min_silence_duration=1,
    dilatation=0.5,
    sample_rate=16000,
    method="auditok",
    avoid_empty_speech=False,
    return_format="tuple",
):
    """
    Remove non-speech segments from audio (using Silero VAD),
    glue the speech segments together and return the result along with
    a function to convert timestamps from the new audio to the original audio

    parameters:
        audio: torch.Tensor
            audio data *in 16kHz*
        use_sample: bool
            if True, return start and end in samples instead of seconds
        min_speech_duration: float
            minimum duration (in sec) of a speech segment
        min_silence_duration: float
            minimum duration (in sec) of a silence segment
        dilatation: float
            how much (in sec) to enlarge each speech segment detected by the VAD
        method: str
            method to use to remove non-speech segments
        avoid_empty_speech: bool
            if True, avoid returning an empty speech segment (re)
    """

    segments = get_vad_segments(
        audio,
        sample_rate=sample_rate,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        method=method,
    )
    segments = apply_dilatation(segments, dilatation, sample_rate, audio, output_sample=True)
    segments = [(seg["start"], seg["end"]) for seg in segments]
    if len(segments) == 0:
        if avoid_empty_speech:
            segments = [(0, audio.shape[-1])]
        else:
            return np.array([]), [], lambda t, t2=None: t if t2 is None else [t, t2]
    segments_s = [
        (float(s) / sample_rate, float(e) / sample_rate) for s, e in segments
    ]
    if not use_sample:
        segments = segments_s
    if return_format == "dict":
        segments = [{"start": s, "end": e} for s, e in segments]
        return None, segments, lambda t, t2=None: do_convert_timestamps(segments_s, t, t2)
    
    audio_speech = np.concatenate([audio[..., s:e] for s, e in segments], axis=-1)
    
    return audio_speech, segments, lambda t, t2=None: do_convert_timestamps(segments_s, t, t2)


def do_convert_timestamps(segments, t, t2=None):
    """
    Convert timestamp from audio without non-speech segments to original audio (with non-speech segments)

    parameters:
        segments: list of tuple (start, end) corresponding to non-speech segments in original audio
        t: timestamp to convert
        t2: second timestamp to convert (optional), when the two timestamps should be in the same segment
    """
    assert len(segments)
    ioffset = 0  # Input offset
    ooffset = 0  # Output offset
    ipreviousend = 0
    result = []
    for istart, iend in segments:
        ostart = ooffset
        oend = ostart + (iend - istart)
        ooffset = oend
        ioffset += istart - ipreviousend
        ipreviousend = iend
        t_in = t <= oend
        t2_in = t_in if t2 is None else t2 <= oend
        if t_in or t2_in:
            result.append(
                [
                    max(istart, min(iend, ioffset + t)),
                    max(istart, min(iend, ioffset + t2)) if t2 is not None else None,
                ]
            )
            if t_in and t2_in:
                break
    if not len(result):
        result.append([ioffset + t, ioffset + t2 if t2 is not None else None])

    if len(result) > 1:
        # Minimize difference between durations
        result = sorted(result, key=lambda x: abs(abs(t2 - t) - abs(x[1] - x[0])))
    result = result[0]
    if t2 is None:
        result = round(result[0], 2)
    else:
        result = [round(x, 2) for x in result]
    return result


def get_vad_segments(
    audio,
    sample_rate=16000,
    min_speech_duration=0.1,
    min_silence_duration=0.1,
    method="auditok",
):
    """
    Get speech segments from audio using the method VAD
    parameters:
        audio: torch.Tensor
            audio data *in 16kHz*
        output_sample: bool
            if True, return start and end in samples instead of seconds
        min_speech_duration: float
            minimum duration (in sec) of a speech segment
        min_silence_duration: float
            minimum duration (in sec) of a silence segment
        dilatation: float
            how much (in sec) to enlarge each speech segment detected by the VAD
        method: str or list
            VAD method to use (auditok, silero, silero:v3.1)
    """
    global _silero_vad_model, _silero_get_speech_ts, _has_onnx
    if isinstance(method, list):
        # Explicit timestamps
        segments = [
            {"start": s * sample_rate, "end": e * sample_rate} for (s, e) in method
        ]
    elif isinstance(method, str) and method.startswith("silero"):
        version = None
        _, version = check_vad_method(method, True)
        # See discussion https://github.com/linto-ai/whisper-timestamped/pull/142/files#r1398326287
        need_folder_hack = version and (version < "v4")

        if _silero_vad_model.get(version) is None:
            # ONNX support since 3.1 in silero
            if (version is None or version >= "v3.1") and (_has_onnx is not False):
                onnx = True
                try:
                    import onnxruntime

                    onnxruntime.set_default_logger_severity(
                        3
                    )  # Remove warning "Removing initializer 'XXX'. It is not used by any node and should be removed from the model."
                    _has_onnx = True
                except ImportError as err:
                    logger.warning(
                        f"Please install onnxruntime to use more efficiently silero VAD"
                    )
                    _has_onnx = False
                    onnx = False
            else:
                onnx = False

            # Choose silero version because of problems with version 4, see  https://github.com/linto-ai/whisper-timestamped/issues/74
            torch_home = os.environ.get("TORCH_HOME", "~/.cache/torch")
            repo_or_dir_master = os.path.expanduser(
                torch_home + "/hub/snakers4_silero-vad_master"
            )
            repo_or_dir_specific = (
                os.path.expanduser(torch_home + f"/hub/snakers4_silero-vad_{version}")
                if version
                else repo_or_dir_master
            )
            repo_or_dir = repo_or_dir_specific
            tmp_folder = None

            def apply_folder_hack():
                nonlocal tmp_folder
                if os.path.exists(repo_or_dir_master):
                    tmp_folder = repo_or_dir_master + ".tmp"
                    shutil.move(repo_or_dir_master, tmp_folder)
                # Make a symlink to the v3.1 model, otherwise it fails
                input_exists = os.path.exists(repo_or_dir_specific)
                if not input_exists:
                    # Make dummy file for the symlink to work
                    os.makedirs(repo_or_dir_specific, exist_ok=True)
                os.symlink(repo_or_dir_specific, repo_or_dir_master)
                if not input_exists:
                    shutil.rmtree(repo_or_dir_specific)

            source = "local"
            if not os.path.exists(repo_or_dir):
                # Load specific version of silero
                repo_or_dir = (
                    f"snakers4/silero-vad:{version}"
                    if version
                    else "snakers4/silero-vad"
                )
                source = "github"
            if need_folder_hack:
                apply_folder_hack()
            try:
                from torch.hub import load as torch_load
                silero_vad_model, utils = torch_load(
                    repo_or_dir=repo_or_dir,
                    model="silero_vad",
                    onnx=onnx,
                    source=source,
                )
                _silero_vad_model[version] = silero_vad_model
            except ImportError as err:
                raise RuntimeError(
                    f"Please install what is needed to use the silero VAD (or use another VAD method)"
                ) from err
            except Exception as err:
                raise RuntimeError(
                    f"Problem when installing silero with version {version}. Check versions here: https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models"
                ) from err
            finally:
                if need_folder_hack:
                    if os.path.exists(repo_or_dir_master):
                        os.remove(repo_or_dir_master)
                    if tmp_folder:
                        shutil.move(tmp_folder, repo_or_dir_master)
            assert os.path.isdir(
                repo_or_dir_specific
            ), f"Unexpected situation: missing {repo_or_dir_specific}"

            _silero_get_speech_ts = utils[0]

        # Cheap normalization of the volume
        
        if isinstance(audio, np.ndarray):
            audio = audio / max(0.1, np.max(np.abs(audio)))
        else:
            audio = audio / max(0.1, audio.abs().max())
        segments = _silero_get_speech_ts(
            audio,
            _silero_vad_model[version],
            sampling_rate=sample_rate,
            min_speech_duration_ms=round(min_speech_duration * 1000),
            min_silence_duration_ms=round(min_silence_duration * 1000),
            return_seconds=False,
        )

    elif method == "auditok":
        # Cheap normalization of the volume
        if isinstance(audio, np.ndarray):
            audio = audio / max(0.1, np.max(np.abs(audio)))
            data = (audio * 32767).astype(np.int16).tobytes()
        else:
            audio = audio / max(0.1, audio.abs().max())
            data = (audio.numpy() * 32767).astype(np.int16).tobytes() 
            
        audio_duration = len(audio) / sample_rate
        import auditok
        segments = auditok.split(
            data,
            sampling_rate=sample_rate,  # sampling frequency in Hz
            channels=1,  # number of channels
            sample_width=2,  # number of bytes per sample
            min_dur=min_speech_duration,  # minimum duration of a valid audio event in seconds
            max_dur=audio_duration,  # maximum duration of an event
            max_silence=min(
                audio_duration * 0.95, min_silence_duration
            ),  # maximum duration of tolerated continuous silence within an event
            energy_threshold=50,
            drop_trailing_silence=True,
        )

        if auditok.__version__ < "0.3.0":
            segments = [
                {"start": s._meta.start * sample_rate, "end": s._meta.end * sample_rate}
                for s in segments
            ]
        else:
            segments = [
                {"start": s.start * sample_rate, "end": s.end * sample_rate}
                for s in segments
            ]

    else:
        raise ValueError(f"Got unexpected VAD method {method}")
    return segments


def apply_dilatation(segments, dilatation, sample_rate, audio, output_sample=False):
    if dilatation > 0:
        dilatation = round(dilatation * sample_rate)
        new_segments = []
        for seg in segments:
            new_seg = {
                "start": max(0, seg["start"] - dilatation),
                "end": min(len(audio), seg["end"] + dilatation),
            }
            if len(new_segments) > 0 and new_segments[-1]["end"] >= new_seg["start"]:
                new_segments[-1]["end"] = new_seg["end"]
            else:
                new_segments.append(new_seg)
        segments = new_segments

    ratio = 1 if output_sample else 1 / sample_rate

    if ratio != 1:
        for seg in segments:
            seg["start"] *= ratio
            seg["end"] *= ratio
    if output_sample:
        for seg in segments:
            seg["start"] = round(seg["start"])
            seg["end"] = round(seg["end"])
    return segments

def check_vad_method(method, with_version=False):
    """
    Check whether the VAD method is valid and return the method in a consistent format

    method: str or list or True or False
    """
    if method in [True, "True", "true"]:
        return check_vad_method("silero")  # default method
    elif method in [None, False, "False", "false", "None", "none"]:
        return None
    elif not isinstance(method, str) and hasattr(method, "__iter__"):
        # list of explicit timestamps
        checked_pairs = []
        for s_e in method:
            assert (
                len(s_e) == 2
            ), f"Got unexpected element {s_e} in the list of VAD segments. Expect (start, end) pairs"
            checked_pairs.append(tuple(s_e))
        return checked_pairs
    elif isinstance(method, str) and method.startswith("silero"):
        version = None
        if method != "silero":
            assert method.startswith("silero:"), f"Got unexpected VAD method {method}"
            version = method.split(":")[1]
            if not version.startswith("v"):
                version = "v" + version
            try:
                assert float(version[1:]) >= 1
            except:
                raise ValueError(
                    f"Got unexpected silero version {version} (please check https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models)"
                )
        if with_version:
            return ("silero", version)
        else:
            return method
    elif method == "auditok":
        try:
            import auditok
        except ImportError:
            raise ImportError(
                "Please install auditok to use the auditok VAD (or use another VAD method)"
            )
    else:
        try:
            method = eval(method)
            assert hasattr(method, "__iter__")
        except:
            raise ValueError(f"Got unexpected VAD method {method}")
        return check_vad_method(method, with_version=with_version)
    return method
