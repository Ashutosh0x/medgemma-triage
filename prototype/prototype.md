# MedGemma CXR Triage - Prototype Summary

## Overview

This prototype demonstrates a chest X-ray urgency triage system using Google's MedGemma 4B multimodal model from the HAI-DEF collection.

## What's Included

### Notebooks
1. **01_quickstart_inference.ipynb** - Basic inference on sample CXR images
2. **02_fine_tune_medgemma.ipynb** - LoRA fine-tuning recipe for urgency classification
3. **03_evaluation_and_metrics.ipynb** - AUC, sensitivity, PPV computation

### Scripts
- `data_scripts/download_datasets.sh` - Dataset access instructions
- `data_scripts/prepare_mimic_subset.py` - MIMIC-CXR data preparation
- `eval/eval_metrics.py` - CLI for reproducible metrics

## Limitations

### Model Limitations
- **Training data bias**: Model trained primarily on US/Western hospital data
- **Rare conditions**: May underperform on rare pathologies not well-represented in training
- **Image quality**: Performance degrades with poor quality or non-standard CXR images
- **Generalization**: Fine-tuned models may not generalize to different populations

### Technical Limitations
- **Memory requirements**: Full precision requires 16GB+ VRAM
- **Inference speed**: ~1-5 seconds per image on GPU
- **ONNX export**: Vision-language models have limited ONNX support

### Evaluation Limitations
- **Dataset size**: Evaluation on limited test set
- **Clinician validation**: Full clinical validation not performed
- **Real-world testing**: Not tested in actual clinical workflow

## Safety Considerations

> **CRITICAL**: This system is for research demonstration only.

1. **Not FDA approved** - Cannot be used for clinical diagnosis
2. **Requires verification** - All outputs need radiologist review
3. **False positives/negatives** - System may miss urgent findings or over-triage
4. **Population bias** - May not perform equally across all populations

## Next Steps for Production

1. **Clinical validation** - Prospective study with radiologist comparison
2. **Regulatory pathway** - FDA 510(k) or De Novo submission
3. **Bias analysis** - Evaluate performance across demographics
4. **Integration testing** - Test with PACS/EHR systems
5. **Monitoring** - Implement drift detection and alerting

## Reproducibility

All experiments can be reproduced using:
- `requirements.txt` for dependencies
- `Dockerfile.prototype` for containerized environment
- Notebooks with fixed random seeds

## Citation

If using this prototype, please cite MedGemma and the datasets used.
See [CITATION.md](../CITATION.md) for details.
