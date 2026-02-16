# Human Evaluation Template

## Overview

This directory contains templates for clinician evaluation of MedGemma outputs.

**Requirement**: n ≥ 10 evaluations for statistical validity.

---

## Evaluation Files

### 1. `evaluation_cases.csv`

Sample cases for clinician review:

| Column | Description |
|--------|-------------|
| `case_id` | Unique identifier |
| `image_path` | Path to CXR image |
| `model_urgency` | Model prediction (Urgent/Non-Urgent) |
| `model_confidence` | Confidence score (0-1) |
| `model_explanation` | Model-generated rationale |
| `model_findings` | Key findings list |

### 2. `clinician_ratings.csv`

To be filled by clinicians:

| Column | Description |
|--------|-------------|
| `case_id` | Matches evaluation_cases.csv |
| `clinician_id` | Anonymous clinician identifier |
| `clinical_rating` | 1-5 scale (see rubric below) |
| `urgency_agree` | Yes/No - does clinician agree with urgency? |
| `explanation_adequate` | Yes/No - is explanation clinically useful? |
| `would_change_decision` | Yes/No - would AI change clinical decision? |
| `comments` | Free text feedback |

---

## Rating Rubric

### Clinical Rating (1-5 Scale)

| Score | Label | Description |
|-------|-------|-------------|
| 5 | Excellent | Perfect urgency, explanation clinically accurate |
| 4 | Good | Correct urgency, explanation mostly accurate |
| 3 | Acceptable | Correct urgency, explanation has minor issues |
| 2 | Poor | Incorrect urgency OR major explanation errors |
| 1 | Unacceptable | Incorrect urgency AND misleading explanation |

### Binary Questions

- **Urgency Agree**: Would you classify this case the same way?
- **Explanation Adequate**: Is the explanation useful for decision-making?
- **Would Change Decision**: Would having this AI output change your clinical workflow?

---

## Summary Metrics

After collecting ratings, calculate:

```python
# Adequacy rate
adequacy_rate = (ratings['clinical_rating'] >= 3).mean()
print(f"Adequacy Rate: {adequacy_rate:.0%}")  # Target: ≥75%

# Mean rating
mean_rating = ratings['clinical_rating'].mean()
print(f"Mean Rating: {mean_rating:.2f}/5.0")  # Target: ≥3.5

# Urgency agreement
urgency_agree = (ratings['urgency_agree'] == 'Yes').mean()
print(f"Urgency Agreement: {urgency_agree:.0%}")

# Inter-rater agreement (if multiple clinicians)
# Calculate Cohen's Kappa or Fleiss' Kappa
```

---

## Sample Cases Template

```csv
case_id,image_path,model_urgency,model_confidence,model_explanation,model_findings
case_001,data/samples/cxr_001.png,Urgent,0.87,"Right lower lobe consolidation consistent with pneumonia","Consolidation in RLL;No pleural effusion;Heart size normal"
case_002,data/samples/cxr_002.png,Non-Urgent,0.23,"Clear lung fields with no acute cardiopulmonary abnormality","Clear lungs;Normal heart size;No effusion"
case_003,data/samples/cxr_003.png,Urgent,0.72,"Cardiomegaly with pulmonary vascular congestion","Enlarged cardiac silhouette;Vascular redistribution;Small bilateral effusions"
```

---

## Instructions for Clinicians

1. Review each case in `evaluation_cases.csv`
2. Look at the CXR image (without AI output first)
3. View the model's prediction and explanation
4. Fill in ratings in `clinician_ratings.csv`
5. Be honest — negative feedback improves the system

**Time estimate**: 2-3 minutes per case

---

## Compliance Notes

- All images are de-identified (no PHI)
- Clinician identities are anonymous (use ID codes)
- This is for research evaluation, not clinical use
- Data use follows MIMIC-CXR/CheXpert agreements
