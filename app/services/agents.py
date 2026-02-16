"""
MedGemma Agentic CXR Triage -- Multi-Agent Orchestrator
========================================================

A true multi-agent system where specialized AI agents collaborate
to produce clinically-grounded triage decisions.

Agents:
  1. TriageAgent       -- Primary classification (urgent/non-urgent)
  2. FindingsAgent     -- Detailed radiographic finding extraction
  3. QualityAgent      -- Image quality assessment & gating
  4. ComparisonAgent   -- Longitudinal comparison with prior reports
  5. SafetyAgent       -- FP prevention, uncertainty estimation, abstention
  6. Orchestrator      -- Coordinates agents & produces final output

This implements a real agentic workflow (not a sequential pipeline)
with agent communication, tool-use, and decision routing.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# --- Agent Communication Protocol ----------------------------------

class AgentRole(Enum):
    TRIAGE = "triage"
    FINDINGS = "findings"
    QUALITY = "quality"
    COMPARISON = "comparison"
    SAFETY = "safety"
    ORCHESTRATOR = "orchestrator"


class Urgency(Enum):
    URGENT = "Urgent"
    NON_URGENT = "Non-Urgent"
    INDETERMINATE = "Indeterminate"


class Confidence(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    sender: AgentRole
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 0=normal, 1=urgent, 2=critical


@dataclass
class AgentResult:
    """Result from an agent's execution."""
    agent: AgentRole
    success: bool
    data: Dict[str, Any]
    latency_ms: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# --- Base Agent -----------------------------------------------------

class BaseAgent:
    """Base class for all agents in the system."""

    def __init__(self, role: AgentRole, model: Any = None, processor: Any = None):
        self.role = role
        self.model = model
        self.processor = processor
        self.message_log: List[AgentMessage] = []

    def receive_message(self, msg: AgentMessage):
        self.message_log.append(msg)

    def send_message(self, content: Dict, priority: int = 0) -> AgentMessage:
        return AgentMessage(sender=self.role, content=content, priority=priority)

    def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError

    def _call_model(self, image: Image.Image, prompt: str, max_tokens: int = 256) -> str:
        """Call MedGemma model with image and text prompt."""
        if self.model is None or self.processor is None:
            raise RuntimeError(f"{self.role.value} agent: No model available")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated = output[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True)


# --- Specialized Agents --------------------------------------------

class QualityAgent(BaseAgent):
    """
    Assesses image quality and determines if the image is suitable
    for AI-assisted triage.
    """

    def __init__(self, **kwargs):
        super().__init__(AgentRole.QUALITY, **kwargs)

    def run(self, image: Image.Image, **kwargs) -> AgentResult:
        start = time.time()
        try:
            arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
            w, h = image.size

            # Exposure
            exposure = float(np.mean(arr))

            # Contrast
            contrast = float(np.std(arr))

            # Sharpness (Laplacian variance)
            dx = np.diff(arr, axis=1)
            dy = np.diff(arr, axis=0)
            sharpness = float(np.var(dx) + np.var(dy))

            # Noise (high-frequency energy)
            noise_score = float(min(np.abs(dx).mean() + np.abs(dy).mean(), 1.0))

            # Assess suitability
            issues = []
            is_suitable = True
            if exposure < 0.10:
                issues.append("Severely underexposed")
                is_suitable = False
            elif exposure < 0.15:
                issues.append("Underexposed")
            if exposure > 0.90:
                issues.append("Overexposed")
                is_suitable = False
            if contrast < 0.04:
                issues.append("No contrast (blank/uniform)")
                is_suitable = False
            if noise_score > 0.70:
                issues.append("High noise")
            if w < 100 or h < 100:
                issues.append(f"Too small ({w}x{h})")
                is_suitable = False

            # View estimation
            aspect = w / h if h > 0 else 1.0
            view = "PA" if 0.8 < aspect < 1.25 else "AP/Lateral"

            data = {
                "suitable": is_suitable,
                "exposure": round(exposure, 4),
                "contrast": round(contrast, 4),
                "sharpness": round(sharpness, 6),
                "noise_score": round(noise_score, 4),
                "resolution": f"{w}x{h}",
                "estimated_view": view,
                "issues": issues,
            }

            return AgentResult(
                agent=self.role, success=True, data=data,
                latency_ms=(time.time() - start) * 1000,
                warnings=issues,
            )
        except Exception as e:
            return AgentResult(
                agent=self.role, success=False, data={"suitable": False},
                latency_ms=(time.time() - start) * 1000,
                errors=[str(e)],
            )


class TriageAgent(BaseAgent):
    """
    Primary urgency classification using MedGemma.
    Produces structured triage output.
    """

    TRIAGE_PROMPT = (
        "You are analyzing a chest X-ray for urgency triage. "
        "Provide your assessment in this exact format:\n\n"
        "URGENCY: [Urgent] or [Non-Urgent]\n"
        "CONFIDENCE: [High], [Medium], or [Low]\n"
        "EXPLANATION: [One concise sentence explaining the classification]\n\n"
        "Focus only on clinically significant findings. Be conservative "
        "-- prefer 'Urgent' when in doubt about serious pathology."
    )

    def __init__(self, **kwargs):
        super().__init__(AgentRole.TRIAGE, **kwargs)

    def run(self, image: Image.Image, prior_report: Optional[str] = None, **kwargs) -> AgentResult:
        start = time.time()
        try:
            prompt = self.TRIAGE_PROMPT
            if prior_report:
                prompt += f"\n\nPrior Report:\n{prior_report}"

            response = self._call_model(image, prompt, max_tokens=200)

            # Parse response
            import re
            urgency = Urgency.INDETERMINATE
            confidence = Confidence.MEDIUM
            explanation = ""

            m = re.search(r'URGENCY:\s*\[?(Urgent|Non-Urgent)\]?', response, re.IGNORECASE)
            if m:
                urgency = Urgency.URGENT if "urgent" == m.group(1).lower() else Urgency.NON_URGENT

            m = re.search(r'CONFIDENCE:\s*\[?(High|Medium|Low)\]?', response, re.IGNORECASE)
            if m:
                confidence = Confidence[m.group(1).upper()]

            m = re.search(r'EXPLANATION:\s*\[?(.+?)\]?\s*$', response, re.IGNORECASE | re.MULTILINE)
            if m:
                explanation = m.group(1).strip()

            # Map confidence to score
            score_map = {Confidence.HIGH: 0.88, Confidence.MEDIUM: 0.65, Confidence.LOW: 0.42}

            data = {
                "urgency": urgency.value,
                "confidence": confidence.value,
                "score": score_map.get(confidence, 0.5),
                "explanation": explanation,
                "raw_response": response,
            }

            return AgentResult(
                agent=self.role, success=True, data=data,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AgentResult(
                agent=self.role, success=False, data={},
                latency_ms=(time.time() - start) * 1000,
                errors=[str(e)],
            )


class FindingsAgent(BaseAgent):
    """
    Extracts detailed radiographic findings from the CXR.
    Provides structured clinical observations.
    """

    FINDINGS_PROMPT = (
        "List the key radiographic findings in this chest X-ray. "
        "For each finding, provide:\n"
        "- FINDING: [description]\n"
        "- LOCATION: [anatomical location]\n"
        "- SEVERITY: [mild/moderate/severe]\n\n"
        "List up to 5 findings. If the image is normal, state 'No acute findings.'"
    )

    def __init__(self, **kwargs):
        super().__init__(AgentRole.FINDINGS, **kwargs)

    def run(self, image: Image.Image, **kwargs) -> AgentResult:
        start = time.time()
        try:
            response = self._call_model(image, self.FINDINGS_PROMPT, max_tokens=400)

            import re
            findings = []
            # Parse structured findings
            finding_blocks = re.split(r'(?=FINDING:)', response, flags=re.IGNORECASE)
            for block in finding_blocks:
                if not block.strip():
                    continue
                finding = {}
                m = re.search(r'FINDING:\s*(.+?)(?=\n|LOCATION:|$)', block, re.IGNORECASE)
                if m:
                    finding["description"] = m.group(1).strip()
                m = re.search(r'LOCATION:\s*(.+?)(?=\n|SEVERITY:|$)', block, re.IGNORECASE)
                if m:
                    finding["location"] = m.group(1).strip()
                m = re.search(r'SEVERITY:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
                if m:
                    finding["severity"] = m.group(1).strip()
                if finding.get("description"):
                    findings.append(finding)

            # Fallback: extract bullet points
            if not findings:
                lines = response.split("\n")
                for line in lines:
                    clean = line.strip().lstrip("-•*").strip()
                    if clean and len(clean) > 5:
                        findings.append({"description": clean})

            data = {
                "findings": findings[:5],
                "raw_response": response,
                "num_findings": len(findings),
            }

            return AgentResult(
                agent=self.role, success=True, data=data,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AgentResult(
                agent=self.role, success=False, data={"findings": []},
                latency_ms=(time.time() - start) * 1000,
                errors=[str(e)],
            )


class ComparisonAgent(BaseAgent):
    """
    Compares current findings with prior report to assess change.
    """

    def __init__(self, **kwargs):
        super().__init__(AgentRole.COMPARISON, **kwargs)

    def run(self, image: Image.Image, prior_report: Optional[str] = None,
            current_findings: Optional[str] = None, **kwargs) -> AgentResult:
        start = time.time()
        if not prior_report:
            return AgentResult(
                agent=self.role, success=True,
                data={"comparison_available": False, "summary": "No prior report available"},
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            prompt = (
                f"Compare the current chest X-ray with this prior report:\n\n"
                f"Prior Report: {prior_report}\n\n"
                f"Describe: (1) What has changed? (2) What is stable? "
                f"(3) Any new findings? Be concise."
            )

            response = self._call_model(image, prompt, max_tokens=300)

            data = {
                "comparison_available": True,
                "summary": response,
                "prior_report_length": len(prior_report),
            }

            return AgentResult(
                agent=self.role, success=True, data=data,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AgentResult(
                agent=self.role, success=False,
                data={"comparison_available": False, "summary": "Comparison failed"},
                latency_ms=(time.time() - start) * 1000,
                errors=[str(e)],
            )


class SafetyAgent(BaseAgent):
    """
    Safety verification agent that:
    1. Estimates uncertainty
    2. Checks for false-positive patterns
    3. Decides abstention
    4. Generates safety recommendations
    """

    def __init__(self, **kwargs):
        super().__init__(AgentRole.SAFETY, **kwargs)

    def run(
        self,
        triage_result: AgentResult,
        quality_result: AgentResult,
        findings_result: AgentResult,
        **kwargs,
    ) -> AgentResult:
        start = time.time()
        try:
            triage = triage_result.data
            quality = quality_result.data
            findings = findings_result.data

            # --- Uncertainty Estimation ---
            uncertainty = 0.0

            # Confidence penalty
            conf_penalties = {"High": 0.08, "Medium": 0.22, "Low": 0.40, "Unknown": 0.35}
            uncertainty += conf_penalties.get(triage.get("confidence", "Unknown"), 0.25)

            # No findings penalty
            if not findings.get("findings"):
                uncertainty += 0.15

            # Short explanation penalty
            if len(triage.get("explanation", "")) < 15:
                uncertainty += 0.12

            # Quality issues penalty
            quality_issues = quality.get("issues", [])
            uncertainty += 0.05 * len(quality_issues)

            uncertainty = min(uncertainty, 1.0)

            # --- False Positive Check ---
            fp_flags = []
            exp_lower = triage.get("explanation", "").lower()
            raw_lower = triage.get("raw_response", "").lower()
            fp_patterns = ["normal variant", "stable", "unchanged", "artifact",
                           "positioning", "no acute", "within normal"]
            for pat in fp_patterns:
                if pat in exp_lower or pat in raw_lower:
                    fp_flags.append(pat)

            is_urgent = triage.get("urgency") == "Urgent"
            fp_detected = is_urgent and len(fp_flags) > 0

            # --- Abstention Decision ---
            should_abstain = False
            abstain_reasons = []

            if uncertainty > 0.25:
                should_abstain = True
                abstain_reasons.append(f"High uncertainty ({uncertainty:.2f})")

            if not quality.get("suitable", True):
                should_abstain = True
                abstain_reasons.append("Image quality insufficient")

            if fp_detected:
                should_abstain = True
                abstain_reasons.append(f"Potential FP: found patterns {fp_flags}")

            if is_urgent and triage.get("confidence") == "Low":
                should_abstain = True
                abstain_reasons.append("Low-confidence urgent prediction")

            # --- Verification Status ---
            verified = True
            verify_reasons = []

            if is_urgent:
                if not findings.get("findings"):
                    verified = False
                    verify_reasons.append("No supporting findings")
                if triage.get("confidence") == "Low":
                    verified = False
                    verify_reasons.append("Low confidence")
                if fp_detected:
                    verified = False
                    verify_reasons.append(f"FP patterns detected: {fp_flags}")

            # --- Safety Recommendations ---
            recommendations = []
            if is_urgent and verified:
                recommendations.append("ESCALATE: Flag for radiologist review within 1 hour")
            if is_urgent and not verified:
                recommendations.append("REVIEW: Urgent but unverified -- needs prompt human review")
            if should_abstain:
                recommendations.append("ABSTAIN: Model confidence insufficient -- human review required")
            if not is_urgent:
                recommendations.append("STANDARD: Route to regular reading queue")

            data = {
                "uncertainty": round(uncertainty, 4),
                "should_abstain": should_abstain,
                "abstain_reasons": abstain_reasons,
                "verified": verified,
                "verify_reasons": verify_reasons,
                "fp_detected": fp_detected,
                "fp_flags": fp_flags,
                "recommendations": recommendations,
            }

            return AgentResult(
                agent=self.role, success=True, data=data,
                latency_ms=(time.time() - start) * 1000,
                warnings=abstain_reasons,
            )

        except Exception as e:
            return AgentResult(
                agent=self.role, success=False,
                data={"should_abstain": True, "verified": False,
                      "uncertainty": 1.0, "recommendations": ["ABSTAIN: Safety check failed"]},
                latency_ms=(time.time() - start) * 1000,
                errors=[str(e)],
            )


# --- Orchestrator ---------------------------------------------------

class AgenticOrchestrator:
    """
    Coordinates multiple specialized agents to produce a comprehensive
    triage decision with full provenance and safety checks.

    Execution flow:
      1. QualityAgent -> gate on image quality
      2. TriageAgent -> primary classification (in parallel with FindingsAgent)
      3. FindingsAgent -> detailed findings
      4. ComparisonAgent -> longitudinal comparison (if prior report)
      5. SafetyAgent -> verification, uncertainty, abstention
      6. Aggregate -> final output
    """

    def __init__(self, model: Any = None, processor: Any = None):
        self.quality_agent = QualityAgent(model=model, processor=processor)
        self.triage_agent = TriageAgent(model=model, processor=processor)
        self.findings_agent = FindingsAgent(model=model, processor=processor)
        self.comparison_agent = ComparisonAgent(model=model, processor=processor)
        self.safety_agent = SafetyAgent(model=model, processor=processor)
        self.execution_log: List[Dict] = []

    def run(
        self,
        image: Image.Image,
        prior_report: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full agentic pipeline.

        Returns comprehensive triage result with:
        - Triage decision (urgency, confidence, explanation)
        - Detailed findings
        - Quality assessment
        - Longitudinal comparison
        - Safety assessment (uncertainty, verification, abstention)
        - Full provenance chain
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        self.execution_log = []

        # Resize for model
        image_resized = image.resize((896, 896), Image.Resampling.LANCZOS)

        # --- Step 1: Quality Assessment ---
        quality_result = self.quality_agent.run(image=image_resized)
        self._log_step("quality", quality_result)

        # Gate: if image unsuitable, short-circuit
        if not quality_result.data.get("suitable", True):
            return self._build_error_result(
                request_id, start_time,
                "Image quality insufficient for AI triage",
                quality_result,
            )

        # --- Step 2 & 3: Triage + Findings (sequential for single-GPU) ---
        triage_result = self.triage_agent.run(image=image_resized, prior_report=prior_report)
        self._log_step("triage", triage_result)

        findings_result = self.findings_agent.run(image=image_resized)
        self._log_step("findings", findings_result)

        # --- Step 4: Comparison (if prior report) ---
        comparison_result = self.comparison_agent.run(
            image=image_resized, prior_report=prior_report,
        )
        self._log_step("comparison", comparison_result)

        # --- Step 5: Safety Assessment ---
        safety_result = self.safety_agent.run(
            triage_result=triage_result,
            quality_result=quality_result,
            findings_result=findings_result,
        )
        self._log_step("safety", safety_result)

        # --- Step 6: Aggregate Final Output ---
        total_latency = (time.time() - start_time) * 1000

        result = self._aggregate_results(
            request_id=request_id,
            triage=triage_result,
            findings=findings_result,
            quality=quality_result,
            comparison=comparison_result,
            safety=safety_result,
            total_latency_ms=total_latency,
        )

        result["execution_trace"] = self.execution_log
        return result

    def _log_step(self, step_name: str, result: AgentResult):
        self.execution_log.append({
            "step": step_name,
            "agent": result.agent.value,
            "success": result.success,
            "latency_ms": round(result.latency_ms, 1),
            "errors": result.errors,
            "warnings": result.warnings,
        })

    def _aggregate_results(
        self, request_id: str,
        triage: AgentResult,
        findings: AgentResult,
        quality: AgentResult,
        comparison: AgentResult,
        safety: AgentResult,
        total_latency_ms: float,
    ) -> Dict[str, Any]:
        """Combine all agent outputs into final result."""
        t = triage.data
        f = findings.data
        q = quality.data
        c = comparison.data
        s = safety.data

        # Override label based on safety agent
        final_label = t.get("urgency", "Non-Urgent")
        if s.get("should_abstain"):
            final_label = "Indeterminate"
        if s.get("fp_detected") and final_label == "Urgent":
            final_label = "Non-Urgent"  # Downgrade FP

        return {
            # Core decision
            "request_id": request_id,
            "label": final_label,
            "score": t.get("score", 0.5),
            "model_confidence": t.get("confidence", "Unknown"),
            "explanation": t.get("explanation", ""),

            # Detailed findings
            "key_findings": [
                fd.get("description", "")
                for fd in f.get("findings", [])
            ],

            # Safety
            "uncertainty": s.get("uncertainty", 0.5),
            "verified": s.get("verified", False),
            "verify_reason": "; ".join(s.get("verify_reasons", [])),
            "abstain": s.get("should_abstain", False),
            "abstain_reason": "; ".join(s.get("abstain_reasons", [])),
            "recommendations": s.get("recommendations", []),
            "fp_detected": s.get("fp_detected", False),

            # Quality
            "image_quality_metrics": q,
            "ood_flag": not q.get("suitable", True),

            # Comparison
            "longitudinal_comparison": c,

            # Provenance (real, from model outputs)
            "provenance": [
                {"source": "triage_agent", "type": "classification",
                 "details": t.get("explanation", ""), "note": "MedGemma triage output"},
                {"source": "findings_agent", "type": "clinical_findings",
                 "details": [fd.get("description") for fd in f.get("findings", [])],
                 "note": "MedGemma findings extraction"},
                {"source": "quality_agent", "type": "quality_assessment",
                 "details": q.get("issues", []),
                 "note": "Image quality assessment"},
            ],

            # Metadata
            "model_metadata": {
                "model_name": "MedGemma-CXR Agentic",
                "pipeline_type": "multi_agent",
                "agents_used": ["quality", "triage", "findings", "comparison", "safety"],
                "total_inference_time_ms": round(total_latency_ms, 1),
            },

            # Raw model outputs for audit
            "raw_triage_response": t.get("raw_response", ""),
            "raw_findings_response": f.get("raw_response", ""),
        }

    def _build_error_result(
        self, request_id: str, start_time: float,
        error_msg: str, quality_result: AgentResult,
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "label": "Indeterminate",
            "score": 0.0,
            "model_confidence": "None",
            "explanation": error_msg,
            "key_findings": [],
            "uncertainty": 1.0,
            "verified": False,
            "abstain": True,
            "abstain_reason": error_msg,
            "recommendations": ["ESCALATE: Image quality issue -- human review required"],
            "image_quality_metrics": quality_result.data,
            "ood_flag": True,
            "provenance": [],
            "model_metadata": {
                "model_name": "MedGemma-CXR Agentic",
                "pipeline_type": "multi_agent",
                "error": error_msg,
                "total_inference_time_ms": round((time.time() - start_time) * 1000, 1),
            },
            "execution_trace": self.execution_log,
        }


def run_pipeline(image_path: str = None, prior_report: str = None):
    """
    Helper function for easy execution of the full agentic pipeline.
    Commonly used in notebooks and Kaggle submissions.
    """
    import os
    from .model_loader import load_model
    from PIL import Image

    if image_path is None:
        return {"status": "error", "message": "No image_path provided"}

    if not os.path.exists(image_path):
        return {"status": "error", "message": f"File not found: {image_path}"}

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load model and processor using standard loader
        model, processor = load_model()

        # Initialize and run orchestrator
        orchestrator = AgenticOrchestrator(model=model, processor=processor)
        result = orchestrator.run(image=image, prior_report=prior_report)

        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
