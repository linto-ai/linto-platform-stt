import io
import os

import numpy as np
import wavio

SAMPLE_RATE = 16000  # whisper.audio.SAMPLE_RATE

import torch
import nemo.collections.asr as nemo_asr
import torchaudio


def has_cuda():
    return torch.cuda.is_available()


def get_device():
    device = os.environ.get("DEVICE", "cuda" if has_cuda() else "cpu")
    use_gpu = "cuda" in device
    try:
        device = torch.device(device)
    except Exception as err:
        raise Exception("Failed to set device: {}".format(str(err))) from err
    return device, use_gpu


def get_language(language = None):
    """
    Get the language from the environment variable LANGUAGE, and format as expected by Whisper.
    """
    if language is None:
        language = os.environ.get("LANGUAGE", None)
    if language is None:
        model = os.environ.get("MODEL", "nvidia/stt_fr_conformer_ctc_large").split("_")
        languages = ["fr", "en", "it", "ru", "ar", "es", "de"]
        for l in languages:
            if l in model:
                language = l
                break
        if language is None:
            language = "en"
    return language

def get_model_class(architecture):
    architecture = architecture.lower()
    model_class = nemo_asr.models.EncDecCTCModelBPE
    if architecture=="ctc" or architecture=="ctc_bpe":
        model_class = nemo_asr.models.EncDecCTCModel
    elif architecture=="hybrid_bpe" or architecture=="rnnt_ctc_bpe" or architecture=="hybrid_rnnt_ctc_bpe":
        model_class = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    elif architecture=="rnnt" or architecture=="rnnt_bpe":
        model_class = nemo_asr.models.EncDecRNNTBPEModel
    elif architecture=="enc_dec" or architecture=="canary" or architecture=="multi_task":
        model_class = nemo_asr.models.EncDecMultiTaskModel
    return model_class
        

def conform_audio(audio, sample_rate=16_000):
    if sample_rate != SAMPLE_RATE:
        # Down or Up sample to the right sampling rate
        audio = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(audio)
    if audio.shape[0] > 1:
        # Stereo to mono
        # audio = torchaudio.transforms.DownmixMono()(audio, channels_first = True)
        audio = audio.mean(0)
    else:
        audio = audio.squeeze(0)
    return audio


def load_audiofile(path):
    if not os.path.isfile(path):
        raise RuntimeError("File not found: %s" % path)
    elif not os.access(path, os.R_OK):
        raise RuntimeError("Missing reading permission for: %s" % path)
    audio, _ = torchaudio.load(path)
    audio = audio.squeeze().numpy()
    return audio


def load_wave_buffer(file_buffer):
    """Formats audio from a wavFile buffer to a torch array for processing."""
    file_buffer_io = io.BytesIO(file_buffer)
    file_content = wavio.read(file_buffer_io)
    sample_rate = file_content.rate
    audio = file_content.data.astype(np.float32) / 32768
    audio = audio.transpose()
    audio = torch.from_numpy(audio)
    return conform_audio(audio, sample_rate).numpy()


def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]