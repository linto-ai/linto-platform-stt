import logging
import torch
import os

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
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
STREAMING_PAUSE_FOR_FINAL=float(os.environ.get("STREAMING_PAUSE_FOR_FINAL", 2.0))
STREAMING_TIMEOUT_FOR_SILENCE=float(os.environ.get("STREAMING_TIMEOUT_FOR_SILENCE", 1.5))   # will consider that silence is detected if no audio is received for the duration of the paquet (dtermined from the first message) * this variable

LONG_FILE_THRESHOLD = int(os.environ.get("LONG_FILE_THRESHOLD", 8*60))  # 8*60=8min
LONG_FILE_CHUNK_LEN = int(os.environ.get("LONG_FILE_CHUNK_LEN", 6*60))  # 6*60=6min. Size of chunk to divide the long file into
LONG_FILE_CHUNK_CONTEXT_LEN = int(os.environ.get("LONG_FILE_CHUNK_CONTEXT_LEN", 10))    # Size of the context of the chunk (10 means 10s before and 10 after)

def set_num_threads(n):
    torch.set_num_threads(n)

NUM_THREADS = os.environ.get("NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
if NUM_THREADS is None:
    NUM_THREADS = torch.get_num_threads()
if NUM_THREADS is not None:
    NUM_THREADS = int(NUM_THREADS)

set_num_threads(1)