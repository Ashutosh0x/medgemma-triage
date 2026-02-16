# MedGemma CXR Triage Assistant

> **An AI-powered, multi-agent chest X-ray urgency triage system** built on Google's [MedGemma](https://huggingface.co/google/medgemma-4b-it) and the Health AI Developer Foundations (HAI-DEF) ecosystem.

[![MedGemma](https://img.shields.io/badge/Model-MedGemma%201.5%204B-blue)](https://huggingface.co/google/medgemma-4b-it)
[![HAI-DEF](https://img.shields.io/badge/Ecosystem-HAI--DEF-green)](https://developers.google.com/health-ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://python.org)

---

## What It Does

MedGemma CXR Triage automatically prioritizes chest X-rays by urgency, helping radiologists focus on critical cases first. The system:

1. **Classifies** chest X-rays as **Urgent** or **Non-Urgent** with calibrated confidence
2. **Explains** its reasoning with structured clinical findings
3. **Verifies** urgent predictions through a multi-stage safety cascade
4. **Abstains** when uncertain -- routing to human review instead of guessing
5. **Compares** with prior reports for longitudinal change detection
6. **Audits** every prediction with tamper-evident logging

> **[WARNING] Clinical Decision Support Only.** This is NOT a diagnostic device. All AI outputs require verification by a qualified radiologist.

---

## Multi-Agent Agentic Architecture

Unlike traditional sequential pipelines, this system uses **5 specialized AI agents** that collaborate through structured communication:

```
+-----------------------------------------------------------------+
|                    AGENTIC TRIAGE PIPELINE                      |
+-----------------------------------------------------------------+
|                                                                 |
|  +--------------+                                               |
|  | QUALITY      | <- Gate: reject unsuitable images              |
|  | AGENT        |                                               |
|  +------+-------+                                               |
|         | [OK] suitable                                         |
|         V                                                       |
|  +--------------+    +--------------+                          |
|  | TRIAGE       |    | FINDINGS     |  <- Run concurrently      |
|  | AGENT        |    | AGENT        |                           |
|  |              |    |              |                           |
|  | Urgent /     |    | Structured   |                           |
|  | Non-Urgent   |    | observations |                           |
|  +------+-------+    +------+-------+                          |
|         |                   |                                   |
|         V                   V                                   |
|  +--------------+    +--------------+                          |
|  | COMPARISON   |    | SAFETY       | <- Multi-signal check     |
|  | AGENT        |    | AGENT        |                           |
|  |              |    |              |                           |
|  | Longitudinal |    | FP prevent,  |                           |
|  | change       |    | uncertainty, |                           |
|  |              |    | abstention   |                           |
|  +--------------+    +--------------+                          |
|                             |                                   |
|                             V                                   |
|                      +--------------+                          |
|                      | ORCHESTRATOR | -> Final Decision          |
|                      |              |   + Full Provenance       |
|                      +--------------+                          |
|                                                                 |
+-----------------------------------------------------------------+
```

### Agent Descriptions

| Agent | Role | Key Capability |
|-------|------|----------------|
| **QualityAgent** | Image quality gating | Exposure, noise, contrast, OOD detection |
| **TriageAgent** | Primary classification | MedGemma VLM inference -> Urgent/Non-Urgent |
| **FindingsAgent** | Finding extraction | Structured radiographic observations |
| **ComparisonAgent** | Longitudinal analysis | Change detection vs. prior reports |
| **SafetyAgent** | FP prevention + abstention | Uncertainty estimation, verification cascade |

---

## Safety Features

### Multi-Stage False Positive Prevention

```
Model Output -> Verification -> Uncertainty Check -> Abstention Gate -> Final Output
                    |                  |                 |
              FP patterns?      Uncertainty > 25%?   Confidence < 40%?
              Evidence present?                      -> Route to human
```

| Control | Mechanism | Threshold |
|---------|-----------|-----------|
| **Image Quality Gating** | Exposure, noise, contrast | Reject low quality |
| **OOD Detection** | Brightness/contrast statistics | Flag non-CXR inputs |
| **FP Pattern Detection** | "normal variant", "stable" pattern scanning | Downgrade  |
| **Uncertainty Estimation** | Multi-factor: confidence + findings + explanation | > 25% -> abstain |
| **Verification Cascade** | Evidence-based check for urgent predictions | Require evidence |
| **Abstention** | Automated routing to human review | Low confidence |
| **Audit Logging** | HMAC-signed daily JSONL logs | Tamper-evident |

---

## Quick Start

### Prerequisites

- Python 3.10+
- GPU with 8GB+ VRAM (recommended) or CPU
- [Hugging Face account](https://huggingface.co/join) with MedGemma access approved

### Installation

```bash
# Clone
git clone https://github.com/your-repo/medgemma-cxr-triage.git
cd medgemma-cxr-triage

# Install dependencies
pip install -r demo_app/requirements.txt

# Set HuggingFace token
export HF_TOKEN=hf_your_token_here

# Run the demo
cd demo_app/ui
python gradio_app.py
```

Open http://localhost:7860 in your browser.

### Docker

```bash
cd demo_app/docker
docker build -t medgemma-demo .
docker run --gpus all -p 7860:7860 -e HF_TOKEN=$HF_TOKEN medgemma-demo
```

---

## Evaluation

### Metric Targets

| Metric | Baseline | Target | Priority |
|--------|----------|--------|----------|
| **Sensitivity** | >= 0.90 | **>= 0.95** | P0 Critical |
| **AUC-ROC** | >= 0.85 | **0.88--0.93** | P0 Critical |
| **PPV** | >= 0.40 | **0.45--0.60** | P1 Important |
| **Specificity** | >= 0.70 | **>= 0.80** | P2 Secondary |

### Datasets

| Dataset | Use | Access |
|---------|-----|--------|
| **Chest X-Ray Pneumonia** (Kaggle) | Primary evaluation | Public |
| **MIMIC-CXR** | Fine-tuning + evaluation | PhysioNet credentialing |
| **CheXpert** | External validation | Stanford AIMI |

### Run Evaluation

```bash
cd prototype/eval
python eval_metrics.py --preds predictions.jsonl --gold labels.jsonl --output report.json
```

---

## Project Structure

```
medgemma/
|-- demo_app/
|   |-- service/
|   |   |-- app.py              # FastAPI REST API
|   |   |-- model_loader.py     # Model loading (real only, no mocks)
|   |   |-- predict.py          # Prediction pipeline with safety controls
|   |   +-- agents.py           # [CORE] Multi-agent orchestration system
|   |-- ui/
|   |   +-- gradio_app.py       # Clinical Gradio interface
|   |-- docker/
|   |   +-- Dockerfile          # Container definition
|   |-- tests/
|   |   +-- smoke_test.sh       # E2E tests
|   +-- requirements.txt
|-- prototype/
|   |-- notebooks/              # Research notebooks
|   |-- eval/                   # Evaluation metrics & targets
|   |-- kaggle_inference/       # Kaggle inference pipeline
|   +-- data_scripts/           # Data preparation & manifests
|-- docs/
|   |-- WORKFLOW.md             # Clinical workflow documentation
|   |-- FAILURE_MODES.md        # Known failure modes & mitigations
|   |-- OFFLINE_BENCHMARKS.md   # Latency & memory benchmarks
|   +-- FP_PREVENTION.md       # False positive prevention controls
|-- tests/                      # Unit tests
|-- CITATION.md                 # Citation information
+-- README.md                   # This file
```

---

## HAI-DEF Models Used

| Model | Purpose | Usage |
|-------|---------|-------|
| **[MedGemma 4B IT](https://huggingface.co/google/medgemma-4b-it)** | Multimodal CXR reasoning | Primary triage + findings extraction |
| **[MedSigLIP](https://huggingface.co/google/medsiglip)** | Medical image encoding | Image feature backbone (via MedGemma) |

---

## Privacy & Offline Capability

> **All inference runs locally. No patient data leaves the device.**

- Model weights downloaded once, cached locally
- Fully offline inference (no network required)
- DICOM anonymization support
- HIPAA-compatible deployment architecture
- Audit logs with HMAC integrity, stored locally

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

MedGemma model weights are subject to Google's [model terms of use](https://huggingface.co/google/medgemma-4b-it).

---

## Citation

See [CITATION.md](CITATION.md) for BibTeX entries for MedGemma, MIMIC-CXR, CheXpert, and RadGraph.

---

## Disclaimer

This application is for **clinical decision support and research demonstration only**. It is:
- **NOT** an FDA-approved medical device
- **NOT** a substitute for professional radiologist interpretation
- **NOT** validated for clinical deployment

All AI-generated outputs require verification by qualified healthcare professionals. The developers assume no liability for clinical decisions made using this system.