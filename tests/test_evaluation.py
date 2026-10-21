"""
Unit Tests for Evaluation Metrics Calculations
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation import calculate_classification_metrics, calculate_multilabel_metrics

def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])  # 7 out of 9 correct
    classes = ["Emergency", "Urgent", "Routine"]

    metrics = calculate_classification_metrics(y_true, y_pred, class_names=classes)
    assert "Accuracy" in metrics
    assert "Sensitivity_Macro" in metrics
    assert "Specificity_Macro" in metrics
    assert "Per_Class_Metrics" in metrics
    assert metrics["Accuracy"] == pytest.approx(7 / 9, 0.01)

def test_multilabel_metrics():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.6], [0.3, 0.1]])
    classes = ["PathologyA", "PathologyB"]

    results = calculate_multilabel_metrics(y_true, y_probs, class_names=classes, threshold=0.5)
    assert "PathologyA" in results
    assert "PathologyB" in results
    assert results["PathologyA"]["Sensitivity"] == 1.0
    assert results["PathologyB"]["Sensitivity"] == 1.0
