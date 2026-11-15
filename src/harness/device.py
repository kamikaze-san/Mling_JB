import torch
import os
import logging

logger = logging.getLogger(__name__)

def resolve_device(requested: str = "auto") -> tuple[str, torch.dtype]:
    """
    Resolve target device and optimal floating point precision across:
    - Windows / Linux NVIDIA GPUs (RTX 4060 8GB laptop, Colab/Kaggle)
    - Apple Silicon Macs (M4 MPS)
    - CPU fallback
    """
    if requested != "auto":
        device = requested.lower()
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Precision resolution
    if device == "cuda":
        # Check bfloat16 support (Ampere/Ada Lovelace e.g. RTX 30xx/40xx)
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info(f"Resolved device: {device}, dtype: {dtype}")
    return device, dtype

def get_quantization_config(device: str, load_in_4bit: bool = True):
    """
    Returns BitsAndBytesConfig for 4-bit loading on CUDA devices with limited VRAM (8GB).
    Returns None for MPS/CPU.
    """
    if device != "cuda" or not load_in_4bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        return bnb_config
    except ImportError:
        logger.warning("bitsandbytes not installed; falling back to unquantized precision.")
        return None

