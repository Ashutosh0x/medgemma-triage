#!/usr/bin/env python3
"""
MedGemma CXR Triage -- Kaggle Inference Pipeline (V10)
=====================================================

Correct, reproducible inference using MedGemma-1.5-4B-IT on the
Chest X-Ray Pneumonia dataset (Kaggle). Outputs structured triage
assessments with real model text and deterministic classification.

Changes from V8:
  - Removed triple-duplicated code blocks
  - Uses AutoModelForImageTextToText (correct for Gemma 3 VLM)
  - Proper chat template with processor.apply_chat_template()
  - Batch inference on BOTH pneumonia and normal test images
  - Real metrics: accuracy, sensitivity, specificity, PPV
  - Structured JSON output per image
"""

import os
import json
import time
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


# ─── Configuration ──────────────────────────────────────────────────
MODEL_SLUG = "google/medgemma/transformers/medgemma-1.5-4b-it/1"
FALLBACK_HF_ID = "google/medgemma-4b-it"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 1  # Per-image (VLM limitation)

# Triage keywords for deterministic classification layer
URGENT_KEYWORDS = [
    "consolidation", "opacity", "effusion", "pneumonia", "infiltrate",
    "infiltration", "edema", "collapse", "mass", "nodule", "pneumothorax",
    "cardiomegaly", "pleural", "atelectasis", "widened mediastinum",
    "abscess", "cavity", "bacteria", "tuberculosis", "fibrosis",
]


# ─── Model Discovery & Loading ─────────────────────────────────────

def discover_model_path() -> Optional[str]:
    """Find MedGemma model weights on Kaggle or download via Hub."""
    # Strategy A: Scan /kaggle/input for config.json
    input_root = Path("/kaggle/input")
    if input_root.exists():
        print("Strategy A: Scanning /kaggle/input for MedGemma config...")
        # Enumerate top-level structure (limited depth)
        for root, dirs, files in os.walk("/kaggle/input"):
            level = root.replace("/kaggle/input", "").count(os.sep)
            if level < 3:
                indent = "  " * level
                print(f"{indent}{os.path.basename(root)}/")
                for f in files[:5]:
                    print(f"{indent}  {f}")

        for p in input_root.rglob("config.json"):
            if "medgemma" in str(p).lower() or "gemma" in str(p).lower():
                print(f"[OK] Found MedGemma config: {p}")
                return str(p.parent)

    # Strategy B: KaggleHub download
    print("Strategy B: Downloading via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.model_download(MODEL_SLUG)
        print(f"[OK] Downloaded model: {path}")
        return path
    except Exception as e:
        print(f"KaggleHub failed: {e}")

    # Strategy C: HuggingFace Hub
    print("Strategy C: Trying HuggingFace Hub...")
    return FALLBACK_HF_ID


def load_model(model_path: str):
    """Load MedGemma model and processor."""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"Loading MedGemma from: {model_path}")
    abs_path = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    is_local = os.path.isdir(abs_path)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if is_local:
        load_kwargs["local_files_only"] = True
        os.environ["HF_HUB_OFFLINE"] = "1"

    processor = AutoProcessor.from_pretrained(
        abs_path, trust_remote_code=True,
        local_files_only=is_local if is_local else False,
    )
    model = AutoModelForImageTextToText.from_pretrained(abs_path, **load_kwargs)

    print(f"[OK] Model loaded | Arch: {model.config.architectures}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Device: {next(model.parameters()).device}")
    return model, processor


# ─── Inference Functions ────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert radiologist assistant for chest X-ray triage. "
    "Analyze images systematically and report findings using "
    "standard radiology terminology."
)

CXR_ANALYSIS_PROMPT = """Analyze this chest X-ray and provide a structured assessment:

1. URGENCY: [Urgent] or [Non-Urgent]
2. CONFIDENCE: [High], [Medium], or [Low]
3. EXPLANATION: A one-sentence summary of the key finding
4. KEY FINDINGS: List 2-3 specific radiographic observations
5. UNCERTAINTY: List any factors reducing confidence

Be precise, use anatomical terms, and express uncertainty where appropriate."""


def build_messages(image: Image.Image, prior_report: Optional[str] = None) -> list:
    """Build chat messages for MedGemma inference."""
    user_content = [
        {"type": "image", "image": image},
    ]
    prompt = CXR_ANALYSIS_PROMPT
    if prior_report:
        prompt += f"\n\nPrior Report:\n{prior_report}"
    user_content.append({"type": "text", "text": prompt})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def run_single_inference(
    model, processor, image: Image.Image,
    prior_report: Optional[str] = None,
) -> Dict:
    """Run inference on a single CXR image and return structured result."""
    messages = build_messages(image, prior_report)

    # Tokenize with proper chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    # Generate
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    # Decode (only new tokens)
    generated = output[0][input_len:]
    response_text = processor.decode(generated, skip_special_tokens=True)

    return {
        "raw_response": response_text,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": int(input_len),
        "output_tokens": int(len(generated)),
    }


def parse_structured_response(text: str) -> Dict:
    """Parse the structured response from MedGemma."""
    import re

    result = {
        "urgency": "Unknown",
        "confidence": "Unknown",
        "explanation": "",
        "key_findings": [],
        "uncertainty": "",
    }

    # Urgency
    m = re.search(r'URGENCY:\s*\[?(Urgent|Non-Urgent)\]?', text, re.IGNORECASE)
    if m:
        result["urgency"] = m.group(1).strip()

    # Confidence
    m = re.search(r'CONFIDENCE:\s*\[?(High|Medium|Low)\]?', text, re.IGNORECASE)
    if m:
        result["confidence"] = m.group(1).strip()

    # Explanation
    m = re.search(r'EXPLANATION:\s*\[?([^\]\n]+)\]?', text, re.IGNORECASE)
    if m:
        result["explanation"] = m.group(1).strip()

    # Key Findings
    m = re.search(
        r'KEY FINDINGS:\s*(.+?)(?=\n\d\.|UNCERTAINTY:|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        findings_text = m.group(1)
        items = re.split(r'[•\-\d\.]+', findings_text)
        result["key_findings"] = [f.strip() for f in items if f.strip()][:5]

    # Uncertainty
    m = re.search(r'UNCERTAINTY:\s*\[?([^\]\n]+)\]?', text, re.IGNORECASE)
    if m:
        result["uncertainty"] = m.group(1).strip()

    return result


def deterministic_triage(description: str) -> Dict:
    """
    Deterministic keyword-based triage layer on top of model output.
    Provides transparent, auditable classification logic.
    """
    text_lower = description.lower()
    hits = [kw for kw in URGENT_KEYWORDS if kw in text_lower]
    is_urgent = len(hits) > 0
    return {
        "triage_label": "URGENT" if is_urgent else "NON-URGENT",
        "trigger_findings": sorted(set(hits)),
        "num_triggers": len(hits),
        "logic": "Keyword detection on MedGemma clinical description",
    }


# --- Dataset Discovery ---------------------------------------------

def discover_test_images(max_per_class: int = 25) -> Tuple[List[Dict], int, int]:
    """Discover chest X-ray test images on Kaggle."""
    images = []
    n_pneumonia = 0
    n_normal = 0

    # Try common Kaggle dataset paths
    search_roots = [
        "/kaggle/input/chest-xray-pneumonia/chest_xray/test",
        "/kaggle/input/chest-xray-pneumonia/chest_xray/val",
    ]

    for root in search_roots:
        if not os.path.exists(root):
            continue

        # Pneumonia
        pneumonia_dir = os.path.join(root, "PNEUMONIA")
        if os.path.isdir(pneumonia_dir):
            files = sorted(glob.glob(os.path.join(pneumonia_dir, "*.jpeg")))
            for f in files[:max_per_class]:
                images.append({"path": f, "ground_truth": "urgent", "class": "PNEUMONIA"})
                n_pneumonia += 1

        # Normal
        normal_dir = os.path.join(root, "NORMAL")
        if os.path.isdir(normal_dir):
            files = sorted(glob.glob(os.path.join(normal_dir, "*.jpeg")))
            for f in files[:max_per_class]:
                images.append({"path": f, "ground_truth": "non-urgent", "class": "NORMAL"})
                n_normal += 1

    print(f"Discovered {len(images)} test images ({n_pneumonia} pneumonia, {n_normal} normal)")
    return images, n_pneumonia, n_normal


# ─── Metrics Computation ───────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute classification metrics from results."""
    tp = fp = tn = fn = 0
    for r in results:
        gt = r["ground_truth"].lower()
        pred = r["predicted_triage"].lower()
        if gt == "urgent":
            if pred == "urgent":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "urgent":
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    return {
        "accuracy": (tp + tn) / total if total else 0,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "ppv": tp / (tp + fp) if (tp + fp) else 0,
        "npv": tn / (tn + fn) if (tn + fn) else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_samples": total,
    }


# ─── Main Entry Point ──────────────────────────────────────────────

def run_kaggle_inference():
    """Main inference pipeline."""
    print(f"[{datetime.now()}] MedGemma CXR Triage -- Kaggle Inference V10")
    print("=" * 60)

    # 1. Discover & load model
    model_path = discover_model_path()
    if not model_path:
        print("CRITICAL: No model found. Aborting.")
        return

    model, processor = load_model(model_path)

    # 2. Discover test images
    test_images, n_pneu, n_norm = discover_test_images(max_per_class=25)
    if not test_images:
        # Fallback: single sample
        fallback = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"
        if os.path.exists(fallback):
            test_images = [{"path": fallback, "ground_truth": "urgent", "class": "PNEUMONIA"}]
        else:
            print("No test images found. Aborting.")
            return

    # 3. Run inference
    results = []
    print(f"\nRunning inference on {len(test_images)} images...")

    for idx, item in enumerate(test_images):
        img_path = item["path"]
        print(f"\n[{idx+1}/{len(test_images)}] {os.path.basename(img_path)} ({item['class']})")

        try:
            img = Image.open(img_path).convert("RGB")
            inference_result = run_single_inference(model, processor, img)

            # Parse structured fields
            parsed = parse_structured_response(inference_result["raw_response"])

            # Deterministic triage overlay
            triage = deterministic_triage(inference_result["raw_response"])

            result_entry = {
                "image": os.path.basename(img_path),
                "ground_truth": item["ground_truth"],
                "ground_truth_class": item["class"],
                "predicted_triage": triage["triage_label"].lower().replace("-", "_").replace("_", "-").replace("non-", "non-"),
                "model_urgency": parsed["urgency"],
                "model_confidence": parsed["confidence"],
                "model_explanation": parsed["explanation"],
                "model_key_findings": parsed["key_findings"],
                "triage_triggers": triage["trigger_findings"],
                "latency_ms": inference_result["latency_ms"],
                "raw_response": inference_result["raw_response"][:500],  # Truncate for storage
            }

            # Normalize prediction label
            pred_lower = triage["triage_label"].lower()
            result_entry["predicted_triage"] = "urgent" if "urgent" in pred_lower and "non" not in pred_lower else "non-urgent"

            results.append(result_entry)
            print(f"  -> Pred: {result_entry['predicted_triage']} | GT: {item['ground_truth']} | Latency: {inference_result['latency_ms']:.0f}ms")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "image": os.path.basename(img_path),
                "ground_truth": item["ground_truth"],
                "predicted_triage": "error",
                "error": str(e),
            })

    # 4. Compute metrics
    valid_results = [r for r in results if r.get("predicted_triage") != "error"]
    metrics = compute_metrics(valid_results) if valid_results else {}

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if metrics:
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV:         {metrics['ppv']:.4f}")
        print(f"  NPV:         {metrics['npv']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"  Confusion:   TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")

    # 5. Save structured output
    output = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "model": "MedGemma-1.5-4B-IT",
        "model_architecture": "Vision-Language Reasoning -> Deterministic Triage",
        "pipeline_version": "V10",
        "num_images": len(test_images),
        "num_successful": len(valid_results),
        "metrics": metrics,
        "results": results,
    }

    output_path = "/kaggle/working/medgemma_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Results saved: {output_path}")

    return output


if __name__ == "__main__":
    run_kaggle_inference()
