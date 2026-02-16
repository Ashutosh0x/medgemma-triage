#!/usr/bin/env python3
"""
Evaluation Metrics CLI for MedGemma CXR Triage
===============================================

Computes AUC-ROC, sensitivity, PPV, and other metrics from prediction files.

Usage:
    python eval_metrics.py --preds predictions.jsonl --gold labels.jsonl
    python eval_metrics.py --preds predictions.jsonl --gold labels.jsonl --output report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load records from a JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    records = []
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
    
    return records


def merge_predictions_and_gold(
    preds: List[Dict],
    gold: List[Dict],
    pred_id_key: str = "id",
    gold_id_key: str = "id"
) -> List[Dict]:
    """Merge predictions with gold labels by ID."""
    gold_by_id = {r[gold_id_key]: r for r in gold}
    
    merged = []
    for pred in preds:
        pred_id = pred.get(pred_id_key)
        if pred_id in gold_by_id:
            merged.append({**gold_by_id[pred_id], **pred})
        else:
            print(f"Warning: No gold label for ID: {pred_id}", file=sys.stderr)
    
    return merged


def extract_labels(
    records: List[Dict],
    gold_label_key: str = "urgency",
    pred_label_key: str = "predicted_urgency",
    score_key: str = "confidence",
    positive_class: str = "urgent"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract binary labels and scores from merged records."""
    y_true = []
    y_pred = []
    y_score = []
    
    for r in records:
        # Gold label
        gold_label = r.get(gold_label_key, "").lower()
        y_true.append(1 if gold_label == positive_class else 0)
        
        # Predicted label
        pred_label = r.get(pred_label_key, "").lower()
        y_pred.append(1 if pred_label == positive_class else 0)
        
        # Score (default to 0.5 if not present)
        score = r.get(score_key, 0.5)
        y_score.append(float(score))
    
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray
) -> Dict:
    """Compute all evaluation metrics."""
    # Basic counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # AUC-ROC (simple trapezoidal approximation)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)
    except ImportError:
        # Fallback: simple AUC calculation
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_true = y_true[sorted_indices]
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            auc = 0.5
        else:
            tpr_sum = 0
            cumsum = 0
            for label in sorted_true:
                if label == 1:
                    cumsum += 1
                else:
                    tpr_sum += cumsum
            auc = tpr_sum / (n_pos * n_neg)
    
    # Sensitivity at 95% recall (approximation)
    if np.sum(y_true) > 0:
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_true = y_true[sorted_indices]
        cumsum = np.cumsum(sorted_true)
        total_positives = np.sum(y_true)
        
        recall_95_idx = np.argmax(cumsum >= 0.95 * total_positives)
        predictions_at_recall_95 = recall_95_idx + 1
        tp_at_recall_95 = cumsum[recall_95_idx]
        sensitivity_at_95_recall = tp_at_recall_95 / predictions_at_recall_95 if predictions_at_recall_95 > 0 else 0.0
    else:
        sensitivity_at_95_recall = 0.0
    
    return {
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true)),
        "auc_roc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "sensitivity_at_95_recall": float(sensitivity_at_95_recall),
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
    }


def print_report(metrics: Dict) -> None:
    """Print metrics report to stdout."""
    print("=" * 50)
    print("MedGemma CXR Triage - Evaluation Report")
    print("=" * 50)
    print(f"Samples:           {metrics['n_samples']}")
    print(f"  Positive:        {metrics['n_positive']}")
    print(f"  Negative:        {metrics['n_negative']}")
    print("-" * 50)
    print(f"AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"Sensitivity:       {metrics['sensitivity']:.4f}")
    print(f"Specificity:       {metrics['specificity']:.4f}")
    print(f"PPV (Precision):   {metrics['ppv']:.4f}")
    print(f"NPV:               {metrics['npv']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Sens. @ 95% Recall:{metrics['sensitivity_at_95_recall']:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TP: {cm['tp']:4d}  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  TN: {cm['tn']:4d}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for MedGemma CXR Triage predictions."
    )
    parser.add_argument(
        "--preds", "-p",
        type=Path,
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--gold", "-g",
        type=Path,
        required=True,
        help="Path to gold labels JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional: output path for JSON report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress stdout output"
    )
    
    args = parser.parse_args()
    
    # Load data
    preds = load_jsonl(args.preds)
    gold = load_jsonl(args.gold)
    
    if not preds:
        print("Error: No predictions found", file=sys.stderr)
        sys.exit(1)
    if not gold:
        print("Error: No gold labels found", file=sys.stderr)
        sys.exit(1)
    
    # Merge and extract labels
    merged = merge_predictions_and_gold(preds, gold)
    if not merged:
        print("Error: No matching records found", file=sys.stderr)
        sys.exit(1)
    
    y_true, y_pred, y_score = extract_labels(merged)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_score)
    
    # Output
    if not args.quiet:
        print_report(metrics)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        if not args.quiet:
            print(f"\nReport saved to: {args.output}")
    
    # Exit with success
    sys.exit(0)


if __name__ == "__main__":
    main()
