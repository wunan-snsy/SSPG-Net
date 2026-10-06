"""Shared E1-compatible metric definitions for all E8 methods.

Detection follows scripts/eval_test_final.py; localization pools TP/FP/FN
over every active image, including all-zero authentic masks.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def detection_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected paired binary labels and continuous scores")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite detection score")
    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pixel_counts(score_map, truth, threshold, *, strict=False):
    score_map = np.asarray(score_map)
    truth = np.asarray(truth) >= 0.5
    if score_map.shape != truth.shape or score_map.shape != (256, 256):
        raise ValueError("Expected aligned 256x256 score map and mask")
    if not np.isfinite(score_map).all():
        raise ValueError("Nonfinite localization map")
    predicted = score_map > threshold if strict else score_map >= threshold
    return {
        "tp": int(np.count_nonzero(predicted & truth)),
        "fp": int(np.count_nonzero(predicted & ~truth)),
        "fn": int(np.count_nonzero(~predicted & truth)),
        "tn": int(np.count_nonzero(~predicted & ~truth)),
    }


def localization_metrics(counts, threshold):
    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
    f1_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    return {
        "pixel_f1": float(2 * tp / f1_den) if f1_den else 1.0,
        "pixel_iou": float(tp / iou_den) if iou_den else 1.0,
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sum_pixel_counts(rows):
    return {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}
