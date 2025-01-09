import logging
import os

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("__stt__")

# The following is to have GPU in the right order (as nvidia-smi show them)
# It is important to set that before loading ctranslate2
# see https://github.com/guillaumekln/faster-whisper/issues/150
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU in the right order

vad = os.environ.get("VAD","auditok").lower()
if vad in ["true", "1"]:
    VAD = "auditok"
elif vad in ["false", "0"]:
    VAD = False
else:
    VAD = os.environ.get("VAD","auditok")

VAD_DILATATION = float(os.environ.get("VAD_DILATATION", 0.5))
VAD_MIN_SPEECH_DURATION = float(os.environ.get("VAD_MIN_SPEECH_DURATION", 0.1))
VAD_MIN_SILENCE_DURATION = float(os.environ.get("VAD_MAX_SILENCE_DURATION", 0.1))

STREAMING_MIN_CHUNK_SIZE=float(os.environ.get("STREAMING_MIN_CHUNK_SIZE", 0.5))
STREAMING_BUFFER_TRIMMING_SEC=float(os.environ.get("STREAMING_BUFFER_TRIMMING_SEC", 8.0))
STREAMING_PAUSE_FOR_FINAL=float(os.environ.get("STREAMING_PAUSE_FOR_FINAL", 1.5))
STREAMING_TIMEOUT_FOR_SILENCE=float(os.environ.get("STREAMING_TIMEOUT_FOR_SILENCE", 1.5))   # will consider that silence is detected if no audio is received for the duration of the paquet (dtermined from the first message) * this variable

try:
    import faster_whisper

    USE_CTRANSLATE2 = True
except ImportError as err:
    try:
        import whisper
    except:
        raise err
    USE_CTRANSLATE2 = False

try:
    import torch

    USE_TORCH = True
except ImportError:
    USE_TORCH = False

try:
    import torchaudio

    USE_TORCHAUDIO = True
except ImportError:
    USE_TORCHAUDIO = False

if USE_CTRANSLATE2:
    def set_num_threads(n):
        # os.environ["OMP_NUM_THREADS"] = str(n)
        pass
    DEFAULT_NUM_THREADS = None
else:
    import torch
    DEFAULT_NUM_THREADS = torch.get_num_threads()
    def set_num_threads(n):
        torch.set_num_threads(n)

# Number of CPU threads
NUM_THREADS = os.environ.get("NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
if NUM_THREADS is None:
    NUM_THREADS = DEFAULT_NUM_THREADS
if NUM_THREADS is not None:
    NUM_THREADS = int(NUM_THREADS)
# For Torch, we will set it afterward, because setting that before loading the model can hang the process (see https://github.com/pytorch/pytorch/issues/58962)
set_num_threads(1)

use_accurate=os.environ.get("USE_ACCURATE","true").lower()
if use_accurate in ["true", "1"]:
    USE_ACCURATE = True
elif use_accurate in ["false", "0"]:
    USE_ACCURATE = False
else:
    raise ValueError(f"USE_ACCURATE must be true, 1, false or 0. Got {os.environ.get('USE_ACCURATE')}")

if USE_ACCURATE:
    DEFAULT_BEAM_SIZE = 5
    DEFAULT_BEST_OF = 5
    DEFAULT_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
else:
    DEFAULT_BEAM_SIZE = None
    DEFAULT_BEST_OF = None
    DEFAULT_TEMPERATURE = 0.0