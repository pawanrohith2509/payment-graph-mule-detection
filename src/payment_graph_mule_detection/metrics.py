"""
Business and Class-Imbalance Metrics for Mule & Fraud Detection.

Computes PR-AUC, ROC-AUC, Recall at fixed precision, and optimal threshold tuning.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    f1_score,
)


@dataclass
class MuleEvaluationResult:
    """Encapsulates comprehensive evaluation metrics under class imbalance."""
    pr_auc: float
    roc_auc: float
    best_f1: float
    best_threshold: float
    recall_at_90_prec: float
    recall_at_80_prec: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "best_f1": round(self.best_f1, 4),
            "best_threshold": round(self.best_threshold, 4),
            "recall_at_90_precision": f"{round(self.recall_at_90_prec * 100, 2)}%",
            "recall_at_80_precision": f"{round(self.recall_at_80_prec * 100, 2)}%",
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tn": self.true_negatives,
        }


def compute_mule_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step_res: int = 100,
) -> MuleEvaluationResult:
    """
    Computes rigorous imbalance metrics given true binary labels and predicted probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # ROC-AUC
    try:
        roc_score = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_score = 0.5

    # PR-AUC
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_score = float(auc(recalls, precisions))

    # Recall at fixed precision levels
    recall_at_90 = 0.0
    recall_at_80 = 0.0

    for p, r in zip(precisions, recalls):
        if p >= 0.90 and r > recall_at_90:
            recall_at_90 = float(r)
        if p >= 0.80 and r > recall_at_80:
            recall_at_80 = float(r)

    # Optimal F1 threshold search
    candidate_thresholds = np.linspace(0.01, 0.99, step_res)
    best_f1 = 0.0
    best_thresh = 0.5

    for th in candidate_thresholds:
        preds = (y_prob >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(th)

    optimal_preds = (y_prob >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, optimal_preds, labels=[0, 1]).ravel()

    return MuleEvaluationResult(
        pr_auc=pr_score,
        roc_auc=roc_score,
        best_f1=best_f1,
        best_threshold=best_thresh,
        recall_at_90_prec=recall_at_90,
        recall_at_80_prec=recall_at_80,
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
    )
