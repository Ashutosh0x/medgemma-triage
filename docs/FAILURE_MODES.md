# MedGemma Failure Modes & Safety Thresholds

## 1. Operating Points & Thresholds

We utilize a conservative cascading threshold system to prioritize safety over automation.

| Metric | Threshold | logic |
|--------|-----------|-------|
| **Urgency Score** | `> 0.50` | Classifies as `Urgent` |
| **High Confidence** | `> 0.70` | Required for automated urgency flagging without manual checks |
| **Uncertainty** | `> 0.25` | Triggers **ABSTENTION** (System refuses to predict) |
| **Noise Score** | `> 0.60` | Rejects image as low quality |
| **Exposure** | `< 0.15` | Rejects image as underexposed |

## 2. Known Failure Modes

### A. The "Silent Failure" (Resolved in V22)
- **Symptom:** Model generates empty string or halts immediately.
- **Cause:** Chat template mismatch or missing `<image_soft_token>` registration.
- **Mitigation:** "Nuclear Bypass" manually injects tokens and forces prefix generation.

### B. Out-of-Distribution (OOD)
- **Scenario:** Non-CXR images (CT scans, limbs, documents) or rotated images.
- **Detection energy:** `detect_ood()` checks brightness statistics.
- **Behavior:** System returns `Indeterminate` and flags `ood_flag=True`.

### C. Hallucination of Specifics
- **Scenario:** Model predicts specific measurements (e.g., "3cm nodule") without evidence.
- **Mitigation:** `verify_prediction` heuristic checks for keywords. We display bounding box ROI as "Approximate Attention" ONLY.

### D. Prior Report Bias
- **Scenario:** If prior report mentions severe disease, model may over-index on it.
- **Mitigation:** `longitudinal_comparison` is displayed separately from current findings.

## 3. Fallback Procedures

In case of system failure or API error:
1. UI displays "Analysis Failed" red banner.
2. `audit` log records the traceback.
3. Clinician must revert to manual standard of care.
