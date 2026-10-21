"""
MediAssist - Chest X-Ray CNN Training Pipeline
Author: Zelvora Industry Labs / MediAssist Team
Trains MobileNetV2 transfer learning model on radiograph cohort.
"""
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import XRAY_CLASSES, XRAY_IMG_SIZE, XRAY_MODEL_PATH, MODELS_DIR
from src.xray_model import build_xray_cnn_model, HAS_TF
from src.utils import generate_anatomical_xray
from src.evaluation import calculate_multilabel_metrics

if not HAS_TF:
    raise ImportError("TensorFlow must be installed to run CNN training.")

import tensorflow as tf
from tensorflow import keras

def create_radiograph_dataset(num_samples_per_class: int = 15):
    """
    Creates an anatomically grounded multi-label chest radiograph dataset.
    """
    images = []
    labels = []

    for class_idx, class_name in enumerate(XRAY_CLASSES):
        for _ in range(num_samples_per_class):
            img_pil = generate_anatomical_xray(class_name.lower(), size=XRAY_IMG_SIZE)
            # Apply slight data augmentation (subtle brightness / contrast)
            img_arr = np.array(img_pil.convert("RGB"), dtype=np.float32)
            # Add subtle random gaussian noise
            noise = np.random.normal(0, 3.0, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255)

            # Create multi-label vector
            label_vec = np.zeros(len(XRAY_CLASSES), dtype=np.float32)
            label_vec[class_idx] = 1.0

            # Simulate co-occurrence (e.g. Cardiomegaly often co-occurs with Effusion)
            if class_name == "Cardiomegaly" and np.random.rand() > 0.6:
                label_vec[XRAY_CLASSES.index("Effusion")] = 1.0
            elif class_name == "Infiltration" and np.random.rand() > 0.6:
                label_vec[XRAY_CLASSES.index("Atelectasis")] = 1.0

            images.append(img_arr)
            labels.append(label_vec)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_and_evaluate_xray(epochs: int = 5, batch_size: int = 8):
    print("=" * 60)
    print("MediAssist: Training MobileNetV2 Chest X-Ray Screener")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Generation
    X, y = create_radiograph_dataset(num_samples_per_class=12)
    print(f"Total synthetic radiograph dataset size: {len(X)} images")

    # Train / Test split (80% train, 20% test)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 2. Build Model
    model = build_xray_cnn_model(num_classes=len(XRAY_CLASSES))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", keras.metrics.AUC(name="auc", multi_label=True)]
    )

    print("\nTraining CNN with MobileNetV2 backbone (transfer learning)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 3. Model Saving
    model.save(str(XRAY_MODEL_PATH))
    print(f"\nModel saved successfully to: {XRAY_MODEL_PATH}")

    # 4. Evaluation on Held-out Test Set
    y_pred_probs = model.predict(X_test, verbose=0)
    per_class_results = calculate_multilabel_metrics(
        y_true=y_test,
        y_pred_probs=y_pred_probs,
        class_names=XRAY_CLASSES,
        threshold=0.40
    )

    print("\nHeld-out Test Set Evaluation (Multi-Label):")
    for cls, met in per_class_results.items():
        print(f"{cls:14s} | Sens: {met['Sensitivity']*100:5.1f}% | Spec: {met['Specificity']*100:5.1f}% | ROC-AUC: {met['ROC_AUC']:.3f} | F1: {met['F1_Score']:.3f}")

    # Save metrics JSON
    metrics_path = MODELS_DIR / "xray_evaluation.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(per_class_results, f, indent=2)
    print(f"\nRadiograph evaluation report written to: {metrics_path}")

    return per_class_results

if __name__ == "__main__":
    train_and_evaluate_xray()
