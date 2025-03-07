import logging
import os

from lockfile import FileLock
from stt import logger, set_num_threads, NUM_THREADS, VAD

from .decoding import decode
from .load_model import load_nemo_model
from .utils import get_device, load_audiofile, load_wave_buffer, get_model_class, get_language

__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]

def warmup():
    model.check_loaded()
    audio = load_audiofile("test/bonjour.wav")
    transcription = decode(audio, MODEL, False)
    logger.info(f"Warmup result: {transcription}")
    
class LazyLoadedModel:
    def __init__(self, model_type, model_class, device, num_threads):
        self.model_type = model_type
        self.model_class = model_class
        self.device = device
        self.num_threads = num_threads
        self._model = None
        self.has_set_num_threads = False

    def check_loaded(self):
        if self._model is None:
            lockfile = os.path.basename(self.model_type)
            with FileLock(lockfile):
                self._model = load_nemo_model(self.model_type, self.model_class, device=self.device)

    def check_num_threads(self):
        if not self.has_set_num_threads and self.num_threads:
            set_num_threads(self.num_threads)
            self.has_set_num_threads = True

    def __getattr__(self, name):
        self.check_loaded()
        self.check_num_threads()
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        self.check_loaded()
        self.check_num_threads()
        return self._model(*args, **kwargs)


# Set informative log
logger.setLevel(logging.INFO)

# Set device
device, USE_GPU = get_device()
logger.info(f"Using device {device}")

# Load ASR model
model_type = os.environ.get("MODEL", "nvidia/stt_fr_conformer_ctc_large")
architecture = get_model_class(os.environ.get("ARCHITECTURE", "ctc_bpe"))

# Check language
language = get_language()
logger.info(f"Using language {language}")

logger.info(f"VAD={VAD}")

logger.info(
    f"Loading Nemo model {model_type} ({'local' if os.path.exists(model_type) else 'remote'})..."
)
try:
    model = LazyLoadedModel(model_type, model_class=architecture, device=device, num_threads=NUM_THREADS)
    MODEL = (model, None)
    if USE_GPU:
        warmup()
except Exception as err:
    raise Exception("Failed to load transcription model: {}".format(str(err))) from err
