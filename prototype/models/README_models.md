# Model Weights Documentation

## Required Models

This project uses the following models from Google's HAI-DEF collection:

### MedGemma 4B IT (Instruction-Tuned)

| Property | Value |
|----------|-------|
| Model ID | `google/medgemma-4b-it` |
| Parameters | 4 billion |
| Type | Multimodal (text + vision) |
| Version | 1.0.1 |
| Created | July 9, 2025 |

**Download:**
```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
```

**Size:** ~8 GB (bfloat16)

### MedSigLIP (Optional - for image-only tasks)

| Property | Value |
|----------|-------|
| Model ID | `google/medsiglip-base-patch16-384` |
| Parameters | 400 million |
| Type | Vision-Language Encoder |

## Fine-tuned Checkpoints

After running fine-tuning (see `02_fine_tune_medgemma.ipynb`), checkpoints are saved to:

```
models/
├── checkpoints/
│   ├── urgency_classifier/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── training_args.json
│   └── explanation_generator/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── training_args.json
└── onnx/
    └── medgemma_quant.onnx
```

## Terms of Use

All models are subject to the [HAI-DEF Terms of Use](https://cloud.google.com/health-ai-developer-foundations/terms).

See [TERMS.md](../../TERMS.md) for compliance requirements.

## Notes

- **Do not redistribute base weights**: Download directly from Hugging Face
- **Fine-tuned adapters only**: Only LoRA/PEFT adapters can be included in this repository
- **Quantized models**: ONNX quantized models are derivatives and may be included
