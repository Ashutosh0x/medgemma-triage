#!/bin/bash
# Entrypoint for MedGemma Demo Container
# =======================================

set -e

echo "=========================================="
echo "MedGemma CXR Triage Demo"
echo "=========================================="

# Print environment info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo ""
echo "Starting application..."
echo "=========================================="

# Default to Gradio app
exec python gradio_app.py "$@"
