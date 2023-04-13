import os
import time

from stt import logger, USE_CTRANSLATE2

if USE_CTRANSLATE2:
    import faster_whisper
else:
    import whisper_timestamped as whisper

def load_whisper_model(model_type_or_file, device="cpu", download_root=None):

    start = time.time()

    logger.info("Loading Whisper model {}...".format(model_type_or_file))

    default_cache_root = os.path.join(os.path.expanduser("~"), ".cache")
    if download_root is None:
        download_root = default_cache_root

    if USE_CTRANSLATE2:
        if not os.path.isdir(model_type_or_file):
            # Note: There is no good way to set the root cache directory
            #       with the current version of faster_whisper:
            #       if "download_root" is specified to faster_whisper.WhisperModel
            #       (or "output_dir" in faster_whisper.utils.download_model),
            #       then files are downloaded directly in it without symbolic links
            #       to the cache directory. So it's different from the behavior
            #       of the huggingface_hub.
            #       So we try to create a symbolic link to the cache directory that will be used by HuggingFace...
            if not os.path.exists(download_root):
                if not os.path.exists(default_cache_root):
                    os.makedirs(download_root)
                    if default_cache_root != download_root:
                        os.symlink(download_root, default_cache_root)
                else:
                    os.symlink(default_cache_root, download_root)
            elif not os.path.exists(default_cache_root):
                os.symlink(download_root, default_cache_root)

        model = faster_whisper.WhisperModel(
            model_type_or_file,
            device=device,
            compute_type="default",
            cpu_threads=0,  # Can be controled with OMP_NUM_THREADS
            num_workers=1,
            # download_root=os.path.join(download_root, f"huggingface/hub/models--guillaumekln--faster-whisper-{model_type_or_file}"),
        )

    else:
        model = whisper.load_model(
            model_type_or_file, device=device,
            download_root=os.path.join(download_root, "whisper")
        )
        model.eval()
        model.requires_grad_(False)

    logger.info("Whisper model loaded. (t={}s)".format(time.time() - start))

    return model