"""
MedGemma CXR Triage -- Comprehensive Test Suite
================================================

Tests for:
  1. Prediction pipeline (parsing, uncertainty, verification, abstention)
  2. Image quality & OOD detection
  3. Multi-agent system (orchestration, inter-agent communication)
  4. Provenance extraction
  5. Audit logging integrity
"""

import hashlib
import hmac
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# Add service directory to path
SERVICE_DIR = Path(__file__).parent.parent / "demo_app" / "service"
sys.path.insert(0, str(SERVICE_DIR))


class TestResponseParsing(unittest.TestCase):
    """Test structured response parsing from model output."""

    def setUp(self):
        from predict import parse_response
        self.parse = parse_response

    def test_parse_urgent_high_confidence(self):
        response = """
        1. URGENCY: [Urgent]
        2. CONFIDENCE: [High]
        3. EXPLANATION: [Large right-sided pleural effusion with mediastinal shift]
        4. KEY FINDINGS: [Pleural effusion; Mediastinal shift; Partial atelectasis]
        5. UNCERTAINTY: [None]
        """
        result = self.parse(response)
        self.assertEqual(result["label"], "Urgent")
        self.assertEqual(result["model_confidence"], "High")
        self.assertAlmostEqual(result["score"], 0.88, places=1)
        self.assertIn("pleural effusion", result["explanation"].lower())
        self.assertTrue(len(result["key_findings"]) >= 1)

    def test_parse_non_urgent(self):
        response = """
        URGENCY: Non-Urgent
        CONFIDENCE: High
        EXPLANATION: Normal chest radiograph with clear lung fields
        """
        result = self.parse(response)
        self.assertEqual(result["label"], "Non-Urgent")

    def test_parse_low_confidence(self):
        response = "URGENCY: Urgent\nCONFIDENCE: Low\nEXPLANATION: Possible opacity"
        result = self.parse(response)
        self.assertEqual(result["model_confidence"], "Low")
        self.assertAlmostEqual(result["score"], 0.42, places=1)

    def test_parse_empty_response(self):
        result = self.parse("")
        self.assertEqual(result["label"], "Non-Urgent")
        self.assertEqual(result["explanation"], "")

    def test_parse_with_prior_report(self):
        response = "URGENCY: Urgent\nCONFIDENCE: Medium\nEXPLANATION: New opacity"
        result = self.parse(response, prior_report="Prior: clear lungs")
        self.assertTrue(result["triage_assessment"]["triage_label"] == "Urgent")
        comp = result.get("longitudinal_comparison", {})
        self.assertTrue(comp.get("prior_available"))


class TestImageQuality(unittest.TestCase):
    """Test image quality assessment and OOD detection."""

    def setUp(self):
        from predict import compute_image_quality_metrics, detect_ood
        self.compute_quality = compute_image_quality_metrics
        self.detect_ood = detect_ood

    def test_normal_image_quality(self):
        # Create a realistic-ish test image
        arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        img = Image.fromarray(arr, "L").convert("RGB")
        metrics = self.compute_quality(img)

        self.assertIn("exposure", metrics)
        self.assertIn("noise_score", metrics)
        self.assertIn("contrast", metrics)
        self.assertIn("sharpness", metrics)
        self.assertIn("resolution", metrics)
        self.assertTrue(0.0 <= metrics["exposure"] <= 1.0)

    def test_black_image_detected_as_ood(self):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        metrics = self.compute_quality(img)
        is_ood, reason = self.detect_ood(img, metrics)
        self.assertTrue(is_ood)
        self.assertIn("underexposed", reason.lower())

    def test_white_image_detected_as_ood(self):
        img = Image.fromarray(np.full((100, 100, 3), 245, dtype=np.uint8))
        metrics = self.compute_quality(img)
        is_ood, reason = self.detect_ood(img, metrics)
        self.assertTrue(is_ood)

    def test_tiny_image_detected(self):
        img = Image.fromarray(np.random.randint(50, 200, (10, 10, 3), dtype=np.uint8))
        metrics = self.compute_quality(img)
        is_ood, reason = self.detect_ood(img, metrics)
        self.assertTrue(is_ood)
        self.assertIn("too_small", reason)

    def test_uniform_image_no_contrast(self):
        img = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
        metrics = self.compute_quality(img)
        is_ood, reason = self.detect_ood(img, metrics)
        self.assertTrue(is_ood)
        self.assertIn("no_contrast", reason)


class TestUncertaintyEstimation(unittest.TestCase):
    """Test uncertainty estimation logic."""

    def setUp(self):
        from predict import estimate_uncertainty
        self.estimate = estimate_uncertainty

    def test_high_confidence_low_uncertainty(self):
        result = {
            "model_confidence": "High",
            "key_findings": ["consolidation"],
            "explanation": "Large right lower lobe consolidation",
            "clinical_findings": {"uncertainty_factors": "None"},
        }
        unc = self.estimate(result)
        self.assertLess(unc, 0.25)

    def test_low_confidence_high_uncertainty(self):
        result = {
            "model_confidence": "Low",
            "key_findings": [],
            "explanation": "",
            "clinical_findings": {"uncertainty_factors": "Poor image quality"},
        }
        unc = self.estimate(result)
        self.assertGreater(unc, 0.5)

    def test_missing_findings_increases_uncertainty(self):
        base = {
            "model_confidence": "Medium",
            "key_findings": ["finding"],
            "explanation": "Some explanation here",
            "clinical_findings": {"uncertainty_factors": ""},
        }
        no_findings = {**base, "key_findings": []}
        unc_with = self.estimate(base)
        unc_without = self.estimate(no_findings)
        self.assertGreater(unc_without, unc_with)


class TestVerification(unittest.TestCase):
    """Test prediction verification logic."""

    def setUp(self):
        from predict import verify_prediction, should_abstain
        self.verify = verify_prediction
        self.abstain = should_abstain

    def test_urgent_with_evidence_passes(self):
        result = {
            "label": "Urgent",
            "model_confidence": "High",
            "key_findings": ["consolidation"],
            "explanation": "Right lower lobe consolidation consistent with pneumonia",
            "image_quality_metrics": {"noise_score": 0.3, "exposure": 0.5},
            "ood_flag": False,
        }
        verified, reason = self.verify(result)
        self.assertTrue(verified)

    def test_urgent_without_findings_fails(self):
        result = {
            "label": "Urgent",
            "model_confidence": "High",
            "key_findings": [],
            "explanation": "Some explanation",
            "image_quality_metrics": {"noise_score": 0.3, "exposure": 0.5},
            "ood_flag": False,
        }
        verified, reason = self.verify(result)
        self.assertFalse(verified)
        self.assertIn("No specific findings", reason)

    def test_urgent_low_confidence_fails(self):
        result = {
            "label": "Urgent",
            "model_confidence": "Low",
            "key_findings": ["opacity"],
            "explanation": "Possible opacity in the lung",
            "image_quality_metrics": {"noise_score": 0.3, "exposure": 0.5},
            "ood_flag": False,
        }
        verified, reason = self.verify(result)
        self.assertFalse(verified)

    def test_non_urgent_bypasses_verification(self):
        result = {
            "label": "Non-Urgent",
            "model_confidence": "High",
            "key_findings": [],
            "explanation": "Normal",
            "image_quality_metrics": {"noise_score": 0.3, "exposure": 0.5},
            "ood_flag": False,
        }
        verified, reason = self.verify(result)
        self.assertTrue(verified)

    def test_fp_pattern_detection(self):
        result = {
            "label": "Urgent",
            "model_confidence": "High",
            "key_findings": ["variant"],
            "explanation": "This is a normal variant, not pathological",
            "image_quality_metrics": {"noise_score": 0.3, "exposure": 0.5},
            "ood_flag": False,
        }
        verified, reason = self.verify(result)
        self.assertFalse(verified)
        self.assertIn("normal variant", reason)

    def test_ood_blocks_verification(self):
        result = {
            "label": "Urgent",
            "ood_flag": True,
            "ood_reason": "underexposed",
            "image_quality_metrics": {},
        }
        verified, reason = self.verify(result)
        self.assertFalse(verified)

    def test_high_uncertainty_triggers_abstention(self):
        result = {"label": "Urgent", "score": 0.6}
        abstain, reason = self.abstain(result, uncertainty=0.30)
        self.assertTrue(abstain)
        self.assertIn("uncertainty", reason.lower())

    def test_low_score_triggers_abstention(self):
        result = {"label": "Urgent", "score": 0.30}
        abstain, reason = self.abstain(result, uncertainty=0.10)
        self.assertTrue(abstain)


class TestProvenance(unittest.TestCase):
    """Test provenance extraction from model output."""

    def setUp(self):
        from predict import extract_provenance_from_response
        self.extract = extract_provenance_from_response

    def test_anatomical_references_extracted(self):
        response = "There is consolidation in the right lower lobe with bilateral effusions."
        prov = self.extract(response, "consolidation in RLL")
        sources = [p["source"] for p in prov]
        self.assertIn("model_anatomical_grounding", sources)
        # Check that actual anatomy was found
        anat = [p for p in prov if p["source"] == "model_anatomical_grounding"][0]
        self.assertTrue(any("bilateral" in d for d in anat["details"]))

    def test_clinical_findings_extracted(self):
        response = "Pneumonia with consolidation and pleural effusion."
        prov = self.extract(response, "pneumonia")
        sources = [p["source"] for p in prov]
        self.assertIn("model_clinical_findings", sources)

    def test_empty_response_still_has_provenance(self):
        prov = self.extract("", "")
        self.assertTrue(len(prov) > 0)
        self.assertEqual(prov[0]["source"], "model_output")

    def test_no_hardcoded_fake_references(self):
        """Ensure no fabricated study IDs or fake dataset references."""
        response = "Normal chest X-ray."
        prov = self.extract(response, "Normal")
        prov_json = json.dumps(prov)
        self.assertNotIn("study_14523", prov_json)
        self.assertNotIn("MIMIC-CXR:study", prov_json)
        self.assertNotIn("hardcoded", prov_json.lower())


class TestAttentionMap(unittest.TestCase):
    """Test attention map computation."""

    def setUp(self):
        from predict import compute_simple_attention_map
        self.compute_attention = compute_simple_attention_map

    def test_returns_structured_result(self):
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = self.compute_attention(img)
        self.assertIn("method", result)
        self.assertIn("top_regions", result)
        self.assertIn("disclaimer", result)
        self.assertNotEqual(result["method"], "none")

    def test_regions_have_pixel_coordinates(self):
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = self.compute_attention(img)
        for region in result.get("top_regions", []):
            px = region.get("pixel_region", {})
            self.assertIn("x", px)
            self.assertIn("y", px)
            self.assertIn("w", px)
            self.assertIn("h", px)

    def test_disclaimer_present(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = self.compute_attention(img)
        self.assertIn("NOT Grad-CAM", result.get("disclaimer", ""))


class TestAuditSignature(unittest.TestCase):
    """Test HMAC audit signature generation."""

    def setUp(self):
        from predict import generate_signature
        self.sign = generate_signature

    def test_signature_format(self):
        data = {"label": "Urgent", "score": 0.85}
        sig = self.sign(data)
        self.assertTrue(sig.startswith("hmac-sha256:"))
        self.assertEqual(len(sig), len("hmac-sha256:") + 32)

    def test_signature_deterministic(self):
        data = {"label": "Urgent", "score": 0.85, "id": "test"}
        sig1 = self.sign(data)
        sig2 = self.sign(data)
        self.assertEqual(sig1, sig2)

    def test_signature_changes_with_data(self):
        sig1 = self.sign({"label": "Urgent"})
        sig2 = self.sign({"label": "Non-Urgent"})
        self.assertNotEqual(sig1, sig2)


class TestAgenticOrchestrator(unittest.TestCase):
    """Test the multi-agent orchestration system."""

    def test_quality_agent_on_good_image(self):
        from agents import QualityAgent
        agent = QualityAgent()
        img = Image.fromarray(np.random.randint(40, 200, (256, 256, 3), dtype=np.uint8))
        result = agent.run(image=img)
        self.assertTrue(result.success)
        self.assertTrue(result.data.get("suitable"))

    def test_quality_agent_on_black_image(self):
        from agents import QualityAgent
        agent = QualityAgent()
        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        result = agent.run(image=img)
        self.assertTrue(result.success)
        self.assertFalse(result.data.get("suitable"))

    def test_safety_agent_fp_detection(self):
        from agents import SafetyAgent, AgentResult, AgentRole

        safety = SafetyAgent()

        triage = AgentResult(
            agent=AgentRole.TRIAGE, success=True,
            data={
                "urgency": "Urgent",
                "confidence": "High",
                "explanation": "This is a normal variant",
                "raw_response": "URGENCY: Urgent... normal variant",
                "score": 0.85,
            },
            latency_ms=100,
        )
        quality = AgentResult(
            agent=AgentRole.QUALITY, success=True,
            data={"suitable": True, "issues": []},
            latency_ms=5,
        )
        findings = AgentResult(
            agent=AgentRole.FINDINGS, success=True,
            data={"findings": [{"description": "variant"}]},
            latency_ms=100,
        )

        result = safety.run(
            triage_result=triage,
            quality_result=quality,
            findings_result=findings,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.data.get("fp_detected"))
        self.assertFalse(result.data.get("verified"))

    def test_safety_agent_high_uncertainty(self):
        from agents import SafetyAgent, AgentResult, AgentRole

        safety = SafetyAgent()
        triage = AgentResult(
            agent=AgentRole.TRIAGE, success=True,
            data={
                "urgency": "Urgent",
                "confidence": "Low",
                "explanation": "",
                "raw_response": "",
                "score": 0.42,
            },
            latency_ms=100,
        )
        quality = AgentResult(
            agent=AgentRole.QUALITY, success=True,
            data={"suitable": True, "issues": ["Underexposed", "High noise"]},
            latency_ms=5,
        )
        findings = AgentResult(
            agent=AgentRole.FINDINGS, success=True,
            data={"findings": []},
            latency_ms=100,
        )

        result = safety.run(
            triage_result=triage,
            quality_result=quality,
            findings_result=findings,
        )

        self.assertTrue(result.data.get("should_abstain"))
        self.assertGreater(result.data.get("uncertainty", 0), 0.25)


class TestTokenization(unittest.TestCase):
    """Tokenization regression tests (from V22 nuclear bypass)."""

    def test_image_token_expansion(self):
        num_patches = 256
        expanded = "<image_soft_token>" * num_patches + "Describe this X-ray."
        self.assertEqual(expanded.count("<image_soft_token>"), 256)
        self.assertTrue(expanded.endswith("Describe this X-ray."))

    def test_chat_template_structure(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi<end_of_turn>"
        )
        output = tokenizer.apply_chat_template([
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi"},
        ])
        self.assertIn("<start_of_turn>", output)
        self.assertIn("user", output)
        self.assertIn("model", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
