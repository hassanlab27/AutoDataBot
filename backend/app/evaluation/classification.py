from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.calibration import calibration_curve

def downsample_curve(
    x: np.ndarray,
    y: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    max_points: int = 100
) -> Tuple[List[float], List[float], List[float]]:
    """Downsamples curve points uniformly for fast, clean Plotly rendering."""
    n = len(x)
    if n <= max_points:
        thresh_list = [round(float(t), 4) for t in thresholds] if thresholds is not None else []
        return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y], thresh_list

    indices = np.linspace(0, n - 1, max_points, dtype=int)
    # Ensure first and last points are preserved
    indices[0] = 0
    indices[-1] = n - 1
    indices = np.unique(indices)

    x_sub = [round(float(x[i]), 4) for i in indices]
    y_sub = [round(float(y[i]), 4) for i in indices]
    thresh_sub = []
    if thresholds is not None and len(thresholds) == n:
        thresh_sub = [round(float(thresholds[i]), 4) for i in indices]
    return x_sub, y_sub, thresh_sub

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    problem_type: str = "binary_classification"
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics, confusion matrices,
    ROC curves, PR curves, threshold exploration, and calibration.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    is_binary = (problem_type == "binary_classification") or (len(np.unique(y_t)) <= 2)

    # Convert classes to string representations for clean JSON
    unique_classes = sorted(list(np.unique(y_t)))
    class_labels = [str(c) for c in unique_classes]

    # 1. Summary Metrics
    accuracy = round(float(accuracy_score(y_t, y_p)), 4)
    balanced_acc = round(float(balanced_accuracy_score(y_t, y_p)), 4)

    pos_label = unique_classes[-1] if is_binary else None
    if is_binary:
        prec = round(float(precision_score(y_t, y_p, pos_label=pos_label, zero_division=0)), 4)
        rec = round(float(recall_score(y_t, y_p, pos_label=pos_label, zero_division=0)), 4)
        f1 = round(float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0)), 4)
    else:
        prec = round(float(precision_score(y_t, y_p, average="macro", zero_division=0)), 4)
        rec = round(float(recall_score(y_t, y_p, average="macro", zero_division=0)), 4)
        f1 = round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4)

    weighted_f1 = round(float(f1_score(y_t, y_p, average="weighted", zero_division=0)), 4)

    metrics_summary: Dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "weighted_f1": weighted_f1,
        "accuracy_pct": round(float(accuracy * 100.0), 1),
        "f1_pct": round(float(f1 * 100.0), 1),
        "precision_pct": round(float(prec * 100.0), 1),
        "recall_pct": round(float(rec * 100.0), 1)
    }

    # 2. Confusion Matrix (Counts and Percentages)
    cm = confusion_matrix(y_t, y_p, labels=unique_classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    correct_cnt = int(np.sum(np.diag(cm)))
    incorrect_cnt = int(len(y_t) - correct_cnt)
    acc_pct = round(float(accuracy * 100.0), 1)
    err_pct = round(float((1.0 - accuracy) * 100.0), 1)

    confusion_dict = {
        "labels": class_labels,
        "matrix": cm.tolist(),
        "normalized_matrix": [[round(float(v), 4) for v in row] for row in cm_norm],
        "matrix_normalized": [[round(float(v), 4) for v in row] for row in cm_norm],
        "total_samples": int(len(y_t)),
        "correct_count": correct_cnt,
        "incorrect_count": incorrect_cnt,
        "accuracy_pct": acc_pct,
        "error_pct": err_pct,
        "summary_text": f"Correctly classified {correct_cnt} of {len(y_t)} test samples ({acc_pct}%), with {incorrect_cnt} mistakes ({err_pct}%).",
        "is_binary": is_binary
    }

    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0
        confusion_dict["binary_breakdown"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": specificity,
            "sensitivity": rec
        }
        metrics_summary["specificity"] = specificity
        metrics_summary["specificity_pct"] = round(float(specificity * 100.0), 1)

    # 3. Per-Class Breakdown
    report_dict = classification_report(y_t, y_p, labels=unique_classes, output_dict=True, zero_division=0)
    per_class: List[Dict[str, Any]] = []
    for c in unique_classes:
        c_str = str(c)
        stats = report_dict.get(c_str, report_dict.get(c, {}))
        per_class.append({
            "class_label": c_str,
            "precision": round(float(stats.get("precision", 0.0)), 4),
            "recall": round(float(stats.get("recall", 0.0)), 4),
            "f1": round(float(stats.get("f1-score", 0.0)), 4),
            "support": int(stats.get("support", 0))
        })

    # 4. Probability-based Curves & Calibration
    roc_data: Optional[Dict[str, Any]] = None
    pr_data: Optional[Dict[str, Any]] = None
    threshold_data: Optional[List[Dict[str, Any]]] = None
    calibration_data: Optional[Dict[str, Any]] = None

    if y_prob is not None:
        try:
            if is_binary:
                # Extract 1D positive probability vector
                prob_vec = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] >= 2) else y_prob.ravel()
                
                # Encode y_true as 0/1 matching unique classes
                y_binary = np.where(y_t == pos_label, 1, 0)

                # ROC Curve
                fpr, tpr, roc_thresh = roc_curve(y_binary, prob_vec)
                roc_auc = round(float(roc_auc_score(y_binary, prob_vec)), 4)
                fpr_sub, tpr_sub, r_thresh_sub = downsample_curve(fpr, tpr, roc_thresh, max_points=100)
                roc_data = {
                    "x": fpr_sub,
                    "y": tpr_sub,
                    "fpr": fpr_sub,
                    "tpr": tpr_sub,
                    "thresholds": r_thresh_sub,
                    "auc": roc_auc
                }
                metrics_summary["roc_auc"] = roc_auc

                # PR Curve
                pr_precision, pr_recall, pr_thresh = precision_recall_curve(y_binary, prob_vec)
                pr_auc = round(float(average_precision_score(y_binary, prob_vec)), 4)
                pr_rec_sub, pr_prec_sub, p_thresh_sub = downsample_curve(pr_recall, pr_precision, pr_thresh, max_points=100)
                pr_data = {
                    "x": pr_rec_sub,
                    "y": pr_prec_sub,
                    "recall": pr_rec_sub,
                    "precision": pr_prec_sub,
                    "thresholds": p_thresh_sub,
                    "average_precision": pr_auc
                }
                metrics_summary["pr_auc"] = pr_auc

                # Threshold Analysis
                thresholds_to_test = np.arange(0.05, 1.0, 0.05)
                thresh_table: List[Dict[str, Any]] = []
                for th in thresholds_to_test:
                    th_round = round(float(th), 2)
                    th_pred = np.where(prob_vec >= th, 1, 0)
                    t_acc = round(float(accuracy_score(y_binary, th_pred)), 4)
                    t_prec = round(float(precision_score(y_binary, th_pred, zero_division=0)), 4)
                    t_rec = round(float(recall_score(y_binary, th_pred, zero_division=0)), 4)
                    t_f1 = round(float(f1_score(y_binary, th_pred, zero_division=0)), 4)
                    th_cm = confusion_matrix(y_binary, th_pred, labels=[0, 1])
                    t_tn, t_fp, t_fn, t_tp = th_cm.ravel()
                    thresh_table.append({
                        "threshold": th_round,
                        "precision": t_prec,
                        "recall": t_rec,
                        "f1": t_f1,
                        "accuracy": t_acc,
                        "true_positives": int(t_tp),
                        "false_positives": int(t_fp),
                        "true_negatives": int(t_tn),
                        "false_negatives": int(t_fn)
                    })
                threshold_data = thresh_table

                # Probability Calibration
                prob_true, prob_pred = calibration_curve(y_binary, prob_vec, n_bins=10, strategy="uniform")
                brier = round(float(brier_score_loss(y_binary, prob_vec)), 4)
                calibration_data = {
                    "brier_score": brier,
                    "prob_predicted": [round(float(v), 4) for v in prob_pred],
                    "prob_pred": [round(float(v), 4) for v in prob_pred],
                    "prob_true": [round(float(v), 4) for v in prob_true]
                }
                metrics_summary["brier_score"] = brier

            else:
                # Multiclass ROC AUC (one-vs-rest)
                if y_prob.ndim == 2 and y_prob.shape[1] == len(unique_classes):
                    multi_auc = round(float(roc_auc_score(y_t, y_prob, multi_class="ovr")), 4)
                    metrics_summary["roc_auc"] = multi_auc
        except Exception:
            pass

    per_class_map = {
        item["class_label"]: {
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
            "support": item["support"]
        } for item in per_class
    }

    return {
        "problem_type": "binary_classification" if is_binary else "multiclass_classification",
        "metrics": metrics_summary,
        "accuracy_pct": f"{metrics_summary.get('accuracy_pct', 0.0):.1f}%",
        "f1_pct": f"{metrics_summary.get('f1_pct', 0.0):.1f}%",
        "precision_pct": f"{metrics_summary.get('precision_pct', 0.0):.1f}%",
        "recall_pct": f"{metrics_summary.get('recall_pct', 0.0):.1f}%",
        "confusion_matrix": confusion_dict,
        "confusion_dict": confusion_dict,
        "per_class": per_class,
        "per_class_metrics": per_class_map,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "threshold_analysis": threshold_data,
        "calibration": calibration_data
    }
