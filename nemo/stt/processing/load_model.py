import os
import shutil
import subprocess
import sys
import time
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceConstants,
    ConfidenceMethodConfig,
    ConfidenceMethodConstants,
)
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

from stt import logger
import logging
logging.basicConfig(level = logging.INFO)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

def load_nemo_model(model_type_or_file, model_class: nemo_asr.models.EncDecHybridRNNTCTCModel, device="cpu", download_root=None):
    start = time.time()
    logger.info(f"Loading Nemo model {model_type_or_file}...")
    default_cache_root = os.path.join(os.path.expanduser("~"), ".cache")
    if download_root is None:
        download_root = default_cache_root
    if model_type_or_file.endswith(".nemo"):
        model = model_class.restore_from(model_type_or_file, map_location=device)
    else:
        model = model_class.from_pretrained(model_type_or_file, map_location=device)
    logger.info(f"Nemo model loaded. (t={time.time() - start:.2f}s)")
    confidence_cfg = ConfidenceConfig(
        preserve_frame_confidence=True, # Internally set to true if preserve_token_confidence == True
        # or preserve_word_confidence == True
        preserve_token_confidence=True, # Internally set to true if preserve_word_confidence == True
        preserve_word_confidence=True,
        aggregation="prod", # How to aggregate frame scores to token scores and token scores to word scores
        exclude_blank=False, # If true, only non-blank emissions contribute to confidence scores
        tdt_include_duration=False, # If true, calculate duration confidence for the TDT models
        method_cfg=ConfidenceMethodConfig( # Config for per-frame scores calculation (before aggregation)
            name="max_prob", # Or "entropy" (default), which usually works better
            entropy_type="gibbs", # Used only for name == "entropy". Recommended: "tsallis" (default) or "renyi"
            alpha=0.5, # Low values (<1) increase sensitivity, high values decrease sensitivity
            entropy_norm="lin" # How to normalize (map to [0,1]) entropy. Default: "exp"
        )
    )
    if isinstance(model, nemo_asr.models.EncDecRNNTModel):
        pass
        # if isinstance(model, nemo_asr.models.EncDecHybridRNNTCTCModel):
        #     model.change_decoding_strategy(decoder_type="ctc")
        #     model.change_decoding_strategy(CTCDecodingConfig(confidence_cfg=confidence_cfg), decoder_type="ctc")
        # else:
        #     model.change_decoding_strategy(RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy_batch", confidence_cfg=confidence_cfg))
    elif isinstance(model, nemo_asr.models.EncDecMultiTaskModel):
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
    else:
        model.change_decoding_strategy(CTCDecodingConfig(confidence_cfg=confidence_cfg))
    return model


def check_torch_installed():
    try:
        import torch
    except ImportError:
        # Install transformers with torch
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers[torch]>=4.23"])

        # # Re-load ctranslate2
        # import importlib
        # import ctranslate2
        # importlib.reload(ctranslate2)
        # importlib.reload(ctranslate2.converters.transformers)

    # import torch

