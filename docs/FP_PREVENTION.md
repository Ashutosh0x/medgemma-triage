# How We Prevent False Positives

## Overview

False positives (FPs) in medical AI are dangerous. This document describes the controls we implement to minimize FPs while maintaining high sensitivity for urgent cases.

---

## 1. Data Provenance Controls

### No Synthetic Data in Evaluation

> **Statement**: No synthetic or mock data are used for reported evaluation metrics. All evaluation datasets are real public datasets; manifests + checksums are included.

### Manifest System

Every training/evaluation file has recorded:
- **SHA256 hash** — File integrity verification
- **Source URL** — Where the data came from
- **Dataset accession** — Which dataset version
- **License/TOU** — Compliance documentation

See: `prototype/data_scripts/build_manifest.py`

### Synthetic File Detection

CI automatically rejects files containing suspicious patterns:
- `synthetic`, `fake`, `mock`, `simulated`
- `generated`, `artificial`, `demo_`

See: `tests/check_no_synthetic.sh`

---

## 2. Model-Level FP Reduction

### Two-Stage Cascade

```
Stage 1: High-sensitivity triage (MedGemma)
    ↓ (if Urgent)
Stage 2: Verification checks
    ↓ (if verified)
Final Output
```

Stage 2 checks:
- At least one key finding present
- Explanation length > 10 chars
- Confidence not "Low"
- No FP patterns in explanation ("normal variant", "stable", "artifact")

### Uncertainty Estimation

```python
uncertainty = base_uncertainty + finding_penalties + explanation_penalties
if uncertainty > 0.25:
    abstain = True  # Human review required
```

Factors that increase uncertainty:
- Low model confidence (+0.35)
- No key findings (+0.15)
- Short explanation (+0.10)
- Explicit uncertainty factors (+0.10 each)

### Abstention Policy

Model abstains and requests human review when:
- Uncertainty > 25%
- Urgent classification with confidence < 70%
- Overall confidence < 40%

---

## 3. Calibration

### Temperature Scaling

Post-hoc calibration on validation set:
```python
T_optimal = minimize(NLL_loss, x0=[1.0])
calibrated_probs = softmax(logits / T_optimal)
```

### Metrics Reported

| Metric | Target | Purpose |
|--------|--------|---------|
| ECE | < 0.10 | Expected Calibration Error |
| Brier | < 0.20 | Probabilistic accuracy |
| Sensitivity | ≥ 95% | Catch urgent cases |
| PPV | ≥ 50% | Limit false positives |

See: `prototype/notebooks/04_baseline_comparison.ipynb`

---

## 4. Evaluation Controls

### External Validation

- Train/val on MIMIC-CXR
- Test on CheXpert (different institution)
- Report both internal and external metrics

### Threshold Selection

```python
# Choose threshold for ≥95% sensitivity
fpr, tpr, thresholds = roc_curve(y_true, y_score)
idx = np.argmax(tpr >= 0.95)
chosen_threshold = thresholds[idx]
```

Always report:
- Sensitivity at chosen threshold
- PPV at chosen threshold
- Number of cases sent to human review

---

## 5. UI Safeguards

### Mandatory Human Confirmation

```
if prediction == "Urgent":
    show "WARNING: Requires clinician verification"
    require confirmation before downstream action
```

### Provenance Display

Every prediction shows:
- Key findings (structured list)
- Confidence level with explanation
- Evidence source (image regions, report snippets)

### Feedback Loop

Clinicians can:
- **Agree** — Logs agreement
- **Disagree** — Logs false positive for retraining
- **Request review** — Escalates to specialist

---

## 6. Audit Trail

All predictions logged to `demo_app/logs/`:

```json
{
  "timestamp": "2026-01-14T12:00:00",
  "request_id": "abc123",
  "label": "Urgent",
  "score": 0.85,
  "uncertainty": 0.15,
  "verified": true,
  "abstained": false,
  "inference_time_ms": 1523
}
```

---

## 7. CI/CD Checks

| Check | Script | Purpose |
|-------|--------|---------|
| No synthetic data | `check_no_synthetic.sh` | Reject mock files |
| Manifest validation | `build_manifest.py --validate` | Verify provenance |
| External eval present | CI check | Require holdout results |
| Threshold documented | README check | Document operating point |

---

## Summary

| Control | Implementation |
|---------|---------------|
| Data provenance | `build_manifest.py` |
| Synthetic detection | `check_no_synthetic.sh` |
| Two-stage cascade | `predict.py` verification |
| Uncertainty estimation | `estimate_uncertainty()` |
| Abstention | `should_abstain()` |
| Calibration | `04_baseline_comparison.ipynb` |
| External validation | Separate test set |
| Audit logging | `logs/*.jsonl` |
| Human confirmation | UI required for urgent |

---

> **This transparency increases trust. Understanding how we prevent false positives is essential for safe deployment.**
