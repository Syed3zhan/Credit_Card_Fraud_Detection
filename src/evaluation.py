"""
Evaluation, PR-AUC Curve & Cost Optimization Module
Calculates metrics, plots curves, and performs cost-sensitive threshold optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def evaluate_model_performance(y_true, y_probs, model_name="Model"):
    """Calculates PR-AUC and ROC-AUC scores for a given model."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_probs)
    return {"Model": model_name, "PR-AUC": pr_auc, "ROC-AUC": roc_auc}


def plot_precision_recall_curve(y_true, y_probs, model_name="XGBoost"):
    """Plots and displays the Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc_score = auc(recall, precision)
    
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc_score:.4f})", color="purple", lw=2)
    plt.xlabel("Recall (True Positive Rate)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("precision_recall_curve.png", dpi=300)
    print("[EVALUATION] Precision-Recall curve saved as 'precision_recall_curve.png'.")
    plt.close()


def optimize_cost_threshold(y_true, y_probs, cost_fn=100, cost_fp=5):
    """Finds decision threshold minimizing business loss.
    
    Cost parameters:
    - Missed Fraud (FN): $100 penalty (average fraud loss)
    - False Alarm (FP): $5 penalty (customer friction cost)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    costs = []
    
    for th in thresholds:
        y_pred = (y_probs >= th).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_0 == 0) & (y_pred == 1)) if 'y_0' in locals() else np.sum((y_true == 0) & (y_pred == 1))
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
        
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    minimum_cost = costs[optimal_idx]
    
    return optimal_threshold, minimum_cost


def calculate_cost_threshold(y_true, y_probs, cost_fn=100, cost_fp=5):
    """Alias function for main.py compatibility."""
    return optimize_cost_threshold(y_true, y_probs, cost_fn, cost_fp)