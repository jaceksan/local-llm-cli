"""Model loading and device detection."""

import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> tuple[str, str]:
    """Detect the best available device and appropriate dtype.

    Returns:
        Tuple of (device, dtype) - device name and recommended dtype.
    """
    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1-M4) - MPS backend
        # float16 is more stable on MPS than bfloat16
        return "mps", "float16"
    else:
        return "cpu", "float32"


def get_device_info(device: str) -> str:
    """Return human-readable device information."""
    if device == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device == "mps":
        return f"Apple Silicon MPS ({platform.processor()})"
    else:
        return "CPU"


def load_model(model_id: str, device: str, dtype: str):
    """Load model and tokenizer.

    Args:
        model_id: Hugging Face model identifier.
        device: Target device (cuda, mps, cpu, or auto).
        dtype: Data type (bfloat16, float16, float32).

    Returns:
        Tuple of (model, tokenizer).
    """
    device_info = get_device_info(device) if device != "auto" else "auto-detect"
    print(f"Loading model: {model_id}")
    print(f"Device: {device_info}, dtype: {dtype}")

    model_kwargs: dict = {"dtype": dtype}

    if device == "auto":
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        model_kwargs["device_map"] = {"": "mps"}
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Model loaded on: {model.device}")
    return model, tokenizer
