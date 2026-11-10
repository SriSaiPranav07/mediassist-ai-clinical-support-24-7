"""
MediAssist - Model Evaluation, Performance Metrics & Boundary Testing
Author: MediAssist Clinical AI Team
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical classification metrics including Specificity and Class-wise Sensitivity.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate specificity and sensitivity per class
    per_class_metrics = {}
    classes = class_names if class_names else [str(i) for i in range(cm.shape[0])]
    
    for i, cls in enumerate(classes):
        if i >= cm.shape[0]:
            break
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1_c = float(2 * (prec * sens) / (prec + sens)) if (prec + sens) > 0 else 0.0

        per_class_metrics[cls] = {
            "Sensitivity_Recall": round(sens, 4),
            "Specificity": round(spec, 4),
            "Precision": round(prec, 4),
            "F1_Score": round(f1_c, 4),
            "Support_TP": int(tp),
            "False_Negatives": int(fn),
            "False_Positives": int(fp),
            "True_Negatives": int(tn)
        }

    # Macro specificity
    macro_spec = float(np.mean([m["Specificity"] for m in per_class_metrics.values()]))

    auc_score = None
    if y_prob is not None:
        try:
            if len(classes) == 2:
                auc_score = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc_score = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc_score = None

    return {
        "Accuracy": round(acc, 4),
        "Sensitivity_Macro": round(rec_macro, 4),
        "Specificity_Macro": round(macro_spec, 4),
        "Precision_Macro": round(prec_macro, 4),
        "F1_Macro": round(f1_macro, 4),
        "ROC_AUC_Macro": round(auc_score, 4) if auc_score is not None else "N/A",
        "Confusion_Matrix": cm.tolist(),
        "Per_Class_Metrics": per_class_metrics,
        "Class_Names": classes
    }

def calculate_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes per-pathology metrics for multi-label radiograph classification.
    """
    results = {}
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    for i, cls in enumerate(class_names):
        true_col = y_true[:, i]
        pred_col = y_pred_binary[:, i]
        prob_col = y_pred_probs[:, i]

        tp = int(np.sum((true_col == 1) & (pred_col == 1)))
        tn = int(np.sum((true_col == 0) & (pred_col == 0)))
        fp = int(np.sum((true_col == 0) & (pred_col == 1)))
        fn = int(np.sum((true_col == 1) & (pred_col == 0)))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * (prec * sens) / (prec + sens)) if (prec + sens) > 0 else 0.0

        try:
            auc = float(roc_auc_score(true_col, prob_col))
        except Exception:
            auc = 0.50

        results[cls] = {
            "Sensitivity": round(sens, 4),
            "Specificity": round(spec, 4),
            "Precision": round(prec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }

    return results
