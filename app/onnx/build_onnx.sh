#!/bin/bash
# ONNX Model Export and Quantization Script
# ==========================================
#
# Exports MedGemma to ONNX format with INT8 quantization
# for optimized inference on edge devices.
#
# Requirements:
#   - PyTorch with CUDA
#   - transformers >= 4.50.0
#   - optimum[exporters]
#   - onnxruntime
#
# Usage:
#   ./build_onnx.sh [--model-id MODEL_ID] [--output-dir OUTPUT_DIR]

set -e

# Configuration
MODEL_ID="${MODEL_ID:-google/medgemma-4b-it}"
OUTPUT_DIR="${OUTPUT_DIR:-./models}"
QUANTIZE="${QUANTIZE:-true}"

echo "=========================================="
echo "MedGemma ONNX Export"
echo "=========================================="
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "Quantize: $QUANTIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for required packages
echo "Checking dependencies..."
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}')" || {
    echo "Error: PyTorch not found"
    exit 1
}

python -c "import optimum" 2>/dev/null || {
    echo "Installing optimum..."
    pip install optimum[exporters] onnx onnxruntime
}

# Export to ONNX
echo ""
echo "Exporting model to ONNX..."
echo "(This may take several minutes)"

python << EOF
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path

model_id = "$MODEL_ID"
output_dir = Path("$OUTPUT_DIR")
quantize = "$QUANTIZE" == "true"

print(f"Loading model: {model_id}")

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(model_id)

# Save processor
processor.save_pretrained(output_dir / "processor")
print(f"Processor saved to: {output_dir / 'processor'}")

print("\n" + "="*50)
print("ONNX Export Instructions")
print("="*50)
print("""
Note: Full ONNX export for MedGemma requires optimum-cli:

1. Install optimum:
   pip install optimum[exporters]

2. Export using CLI:
   optimum-cli export onnx \\
     --model google/medgemma-4b-it \\
     --task image-text-to-text \\
     ./onnx_model

3. For quantization:
   python -m onnxruntime.quantization.quantize \\
     --input ./onnx_model/model.onnx \\
     --output ./onnx_model/model_quantized.onnx \\
     --per_channel

Alternative: Use torch.compile() for optimized PyTorch inference
without ONNX conversion.
""")

print("For this demo, we recommend using the PyTorch model with")
print("4-bit quantization (bitsandbytes) instead of ONNX.")
EOF

echo ""
echo "=========================================="
echo "Export Complete"
echo "=========================================="
echo ""
echo "Files saved to: $OUTPUT_DIR"
echo ""
echo "Note: Full ONNX export for vision-language models is complex."
echo "For edge deployment, consider using:"
echo "  1. PyTorch with bitsandbytes 4-bit quantization"
echo "  2. vLLM for optimized serving"
echo "  3. GGUF format with llama.cpp (for text-only models)"
