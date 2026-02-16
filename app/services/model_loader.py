"""
MedGemma Model Loader (V3.0)
=============================

Loads MedGemma VLM from Hugging Face Hub or local cache.
Supports:
  - BFloat16 (default, GPU)
  - 4-bit quantization (via bitsandbytes)
  - CPU fallback
  - Offline / air-gapped deployment

NO mock models. If model cannot load, raises a clear error.
"""

import os
import sys
from typing import Any, Optional, Tuple

import torch

# ─── Configuration via Environment ──────────────────────────────────
MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
USE_QUANTIZATION = os.environ.get("MEDGEMMA_QUANTIZE", "true").lower() == "true"
DEVICE = os.environ.get("MEDGEMMA_DEVICE", "auto")
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def print_device_info() -> None:
    """Print available compute device information."""
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  Device config:   {DEVICE}")
    print(f"  Quantization:    {USE_QUANTIZATION}")
    print(f"  Model ID:        {MODEL_ID}")
    print("=" * 50)


def load_model(
    model_id: str = MODEL_ID,
    use_quantization: bool = USE_QUANTIZATION,
    device: str = DEVICE,
) -> Tuple[Any, Any]:
    """
    Load MedGemma model and processor.

    Args:
        model_id: HF model ID or local path
        use_quantization: Whether to use 4-bit quantization
        device: Device mapping ('auto', 'cuda', 'cpu')

    Returns:
        (model, processor) tuple

    Raises:
        RuntimeError: If model cannot be loaded
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print_device_info()

    # Determine dtype and quantization config
    quantization_config = None
    torch_dtype = torch.bfloat16

    has_cuda = torch.cuda.is_available()

    if use_quantization and has_cuda:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = torch.bfloat16
            print("[OK] Using 4-bit NF4 quantization")
        except ImportError:
            print("[WARN] bitsandbytes not available -- using bfloat16")
            quantization_config = None

    if not has_cuda:
        torch_dtype = torch.float32
        device = "cpu"
        print("[WARN] No CUDA GPU -- using CPU with float32 (slower)")
        if quantization_config:
            print("  Disabling quantization on CPU")
            quantization_config = None

    # Resolve device_map
    device_map = "auto" if has_cuda and device == "auto" else None

    # Load model
    is_local = os.path.isdir(model_id)
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
    if device_map:
        load_kwargs["device_map"] = device_map
    if is_local:
        load_kwargs["local_files_only"] = True
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    print(f"\nLoading model: {model_id}")
    print(f"  Local: {is_local}")
    print(f"  Dtype: {torch_dtype}")
    print(f"  Device map: {device_map}")
    print(f"  Quantized: {quantization_config is not None}")

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        if device == "cpu" and not quantization_config:
            model = model.to("cpu")

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=is_local,
            token=HF_TOKEN if HF_TOKEN else None,
        )

        # Report success
        param_count = sum(p.numel() for p in model.parameters())
        param_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"\n[OK] Model loaded successfully")
        print(f"  Parameters: {param_count:,}")
        print(f"  Memory: ~{param_size_gb:.2f} GB")
        print(f"  Architecture: {model.config.architectures}")

        return model, processor

    except Exception as e:
        error_msg = f"Failed to load MedGemma model '{model_id}': {e}"
        print(f"\n[FAIL] {error_msg}", file=sys.stderr)

        # Provide actionable guidance
        if "401" in str(e) or "gated" in str(e).lower():
            print("\n  [->] Model requires HuggingFace access approval.")
            print("    1. Accept terms at: https://huggingface.co/google/medgemma-4b-it")
            print("    2. Set HF_TOKEN environment variable")
        elif "disk space" in str(e).lower() or "no space" in str(e).lower():
            print("\n  [->] Insufficient disk space. Model requires ~8-16 GB.")
        elif "out of memory" in str(e).lower():
            print("\n  [->] Insufficient GPU memory. Try:")
            print("    export MEDGEMMA_QUANTIZE=true")
            print("    export MEDGEMMA_DEVICE=cpu")

        raise RuntimeError(error_msg) from e


def check_model_health(model: Any, processor: Any) -> bool:
    """Quick health check: verify model can tokenize and generate."""
    try:
        test_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello, what is your name?"}
            ]}
        ]
        inputs = processor.apply_chat_template(
            test_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        return output.shape[-1] > inputs["input_ids"].shape[-1]
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing model loader...")
    try:
        model, processor = load_model()
        healthy = check_model_health(model, processor)
        print(f"\nHealth check: {'[OK] PASS' if healthy else '[FAIL] FAIL'}")
    except RuntimeError as e:
        print(f"Model not available: {e}", file=sys.stderr)
        sys.exit(1)
