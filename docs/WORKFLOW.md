# MedGemma Agentic Clinical Workflow

## Multi-Agent CXR Triage Pipeline

```
+-----------------------------------------------------------------+
|                    AGENTIC TRIAGE PIPELINE                      |
+-----------------------------------------------------------------+
|                                                                 |
|         [Image + Prior Report]                                  |
|                |                                                |
|                V                                                |
|  +----------------------+                                       |
|  |  [QUALITY AGENT]    | Step 1: Image Quality Gating         |
|  |                      |                                      |
|  |  * Exposure check    | reject -> INDETERMINATE               |
|  |  * Noise estimation  |         + "Human review required"     |
|  |  * Contrast check    |                                      |
|  |  * OOD detection     |                                      |
|  +----------+-----------+                                      |
|             | [OK] passes quality                                  |
|             V                                                   |
|  +----------------------+    +----------------------+         |
|  |  [TRIAGE AGENT]     |    |  [FINDINGS AGENT]    |         |
|  |                      |    |                      |          |
|  |  MedGemma 4B VLM     |    |  MedGemma 4B VLM     |         |
|  |  -> Urgent/Non-Urgent |    |  -> Structured list    |         |
|  |  -> Confidence level  |    |  -> Location + severity|         |
|  |  -> Explanation       |    |                      |          |
|  +----------+-----------+    +----------+-----------+         |
|             |                           |                      |
|             V                           |                      |
|  +----------------------+              |                      |
|  |  [COMPARISON AGENT] |              |                      |
|  |  (if prior report)   |              |                      |
|  |                      |              |                      |
|  |  MedGemma 4B VLM     |              |                      |
|  |  -> Change detection  |              |                      |
|  |  -> Progression/      |              |                      |
|  |    regression        |              |                      |
|  +----------+-----------+              |                      |
|             |                           |                      |
|             V                           V                      |
|  +----------------------------------------------+             |
|  |  [SAFETY AGENT]                              |             |
|  |                                                |             |
|  |  * Uncertainty estimation (multi-factor)       |             |
|  |  * False positive pattern detection            |             |
|  |  * Evidence verification (findings -> urgent?)  |             |
|  |  * Abstention decision (confidence < 40%?)     |             |
|  |  * Clinical recommendations                    |             |
|  +----------------------+-------------------------+             |
|                         |                                       |
|                         V                                       |
|  +----------------------------------------------+             |
|  |  [ORCHESTRATOR] (Aggregation)                |             |
|  |                                                |             |
|  |  -> Final triage label (Urgent/Non-Urgent/      |             |
|  |    Indeterminate)                               |             |
|  |  -> Provenance chain (every agent's output)     |             |
|  |  -> Audit log entry (HMAC-signed)               |             |
|  |  -> UI display data                              |             |
|  +------------------------------------------------+             |
|                                                                  |
+-----------------------------------------------------------------+
```

## Pipeline Steps (Detailed)

### Step 1: Quality Agent
- **Input**: Raw CXR image
- **Checks**: Exposure (min 0.15), noise (<0.70), contrast (>0.04), resolution (>100px)
- **Output**: `suitable: bool`, quality metrics
- **Gate**: If unsuitable -> skip all agents -> return `Indeterminate`

### Step 2: Triage Agent (MedGemma 4B)
- **Input**: CXR image + optional prior report context
- **Prompt**: Structured triage prompt requesting URGENCY, CONFIDENCE, EXPLANATION
- **Output**: Classification label, confidence level, clinical reasoning
- **Model**: `google/medgemma-4b-it` (or `medgemma-1.5-4b-it`)

### Step 3: Findings Agent (MedGemma 4B)
- **Input**: CXR image
- **Prompt**: Structured findings extraction (FINDING, LOCATION, SEVERITY)
- **Output**: Up to 5 structured radiographic observations
- **Note**: Runs after triage agent (single GPU constraint)

### Step 4: Comparison Agent (MedGemma 4B)
- **Input**: CXR image + prior report text
- **Prompt**: Change detection vs. prior
- **Output**: Summary of changes, stable findings, new findings
- **Skip**: If no prior report provided

### Step 5: Safety Agent
- **Input**: All previous agent outputs
- **Logic**:
  1. **Uncertainty estimation** -- Weighted sum of:
     - Confidence penalty (Low=+0.40, Medium=+0.22, High=+0.08)
     - No findings penalty (+0.15)
     - Short explanation penalty (+0.12)
     - Quality issues penalty (+0.05 each)
  2. **FP pattern detection** — Scan explanation for "normal variant", "stable", etc.
  3. **Verification** — If urgent: require ≥1 finding, explanation >10 chars
  4. **Abstention** — If uncertainty >25% or urgent with low confidence
- **Output**: uncertainty, abstention decision, verification status, recommendations

### Step 6: Orchestrator
- Aggregates all agent outputs
- Applies safety overrides (FP → downgrade, abstention → Indeterminate)
- Generates provenance chain linking each agent's contribution
- Produces HMAC-signed audit log entry
- Returns structured result for API/UI

---

## Decision Matrix

| Confidence | Urgency | Safety Agent | Action |
|------------|---------|-------------|--------|
| High | Urgent | Verified | **Immediate escalation** |
| Medium | Urgent | Verified | Priority queue + flag |
| Low | Urgent | Not verified | **ABSTAIN** → human review |
| Any | Urgent | FP detected | Downgrade to Non-Urgent |
| High | Non-Urgent | — | Standard queue |
| Medium | Non-Urgent | — | Standard queue |
| Low | Non-Urgent | — | **ABSTAIN** → human review |
| — | Indeterminate | — | **ABSTAIN** → human review |

---

## Privacy & Offline

> **All inference runs locally. No patient data leaves the device.**

- Model weights downloaded once from HuggingFace Hub
- Zero network calls during inference
- HIPAA-compatible: no PHI in logs or model I/O
- Audit logs stored locally with HMAC integrity

---

## Alignment with HAI-DEF Goals

| HAI-DEF Principle | Implementation |
|-------------------|----------------|
| Human-centered | 5-agent system with human always in the loop |
| Safety-first | Multi-stage verification + uncertainty-driven abstention |
| Accessible | Offline-capable, 4-bit quantization for edge devices |
| Transparent | Full agent execution trace + provenance chain |
| Verifiable | HMAC-signed audit logs for every prediction |
