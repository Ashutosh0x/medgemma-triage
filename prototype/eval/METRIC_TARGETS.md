# MedGemma Evaluation Targets & Benchmarks

## Judge-Friendly Metric Targets

Based on MedGemma 1.5 release claims and realistic expectations for a fine-tuned 4B model.

### Interpreting Release Claims

| Claim | Type | Calculation |
|-------|------|-------------|
| "22% improvement" on EHR QA | **Relative** | new = baseline × 1.22 |
| "58% fewer errors" on CXR dictation | **Relative WER reduction** | new_WER = baseline_WER × (1 - 0.58) |

**Example**: If baseline MedQA = 64.4%, then 22% relative improvement -> 64.4 x 1.22 = **78.57%**

---

## Classification / Triage Targets

| Metric | Baseline (acceptable) | Target (competitive) | Priority |
|--------|----------------------|---------------------|----------|
| **Sensitivity** (urgent cases) | >= 0.90 | **>= 0.95** | P0 Critical |
| **AUC-ROC** | >= 0.85 | **0.88 - 0.93** | P0 Critical |
| **PPV** (at high-recall point) | >= 0.40 | **0.45 - 0.60** | P1 Important |
| **Specificity** | >= 0.70 | >= 0.80 | P2 Secondary |

### Operating Point Selection
```
Choose threshold where: Sensitivity >= 0.95
Report: PPV at that threshold
```

---

## Localization / Explainability Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **IoU** (bounding boxes vs. lesion annotations) | ≥ 0.35 – 0.50 | Hard for diffuse findings |
| **Hit-rate@3** (top-3 crops include lesion) | ≥ 0.80 | For urgent cases |
| **Provenance coverage** | 100% | Every urgent prediction has ≥1 evidence |

---

## Report Generation / Extraction Targets

| Metric | Target | Benchmark |
|--------|--------|-----------|
| **RadGraph F1** (entity extraction) | ≥ 0.75 – 0.85 | Key findings & anatomy |
| **Clinical adequacy** (human rating) | ≥ 75% adequate | n ≥ 10 clinicians |
| **Mean rating** | ≥ 3.5 / 5.0 | 5-point Likert scale |

---

## MedASR Speech Targets

| Metric | Baseline (generalist ASR) | Target (MedASR) | Improvement |
|--------|--------------------------|-----------------|-------------|
| **WER** (CXR dictation) | ~20% | **≤ 8.4%** | 58% relative reduction |
| **WER** (specialized medical) | ~15% | **≤ 2.7%** | 82% relative reduction |

**Reporting format**:
```
"WER dropped from 20.0% → 8.4% (absolute −11.6 pp, relative −58%)"
```

---

## System / UX Targets

| Metric | Target | Device |
|--------|--------|--------|
| **Inference latency** | ≤ 2s per image | M1 laptop / RTX 3090 |
| **Inference latency** | ≤ 5s per image | Jetson / mobile |
| **Model size** (quantized) | ≤ 4–8 GB | ONNX INT8 on disk |
| **RAM usage** | ≤ 12 GB | During inference |

---

## Datasets & Benchmarks

### Required Datasets
| Dataset | Source | Use |
|---------|--------|-----|
| **MIMIC-CXR** | PhysioNet | Image triage, report generation |
| **CheXpert** | Stanford AIMI | Classification labels |
| **Chest ImaGenome** | PhysioNet | Localization annotations |
| **MedQA** | Public | EHR QA evaluation |
| **AfriMed-QA** | Public | Diverse clinical QA |

### Evaluation Tools
- **Metrics**: scikit-learn, jiwer (WER), RadGraph
- **Clinician rating**: CSV template + mean/% adequate calculation
- **Latency**: Python `time` + NVIDIA nsight for profiling

---

## Writeup Checklist

- [ ] Report **both absolute and relative** changes for all comparisons
- [ ] State exact **dataset, split, and preprocessing** steps
- [ ] Include **n ≥ 10 clinician ratings** with mean and % adequate
- [ ] Show **runtime screenshots** with latency and memory
- [ ] Include **quantized model size** in MB/GB
- [ ] Clarify **TOU/IRB compliance** for each dataset
