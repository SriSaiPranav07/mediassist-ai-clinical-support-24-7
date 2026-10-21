# MediAssist Models Directory

This directory stores trained model weights, vectorizer artifacts, and evaluation reports.

## Model Checkpoints
- `triage_classifier.joblib`: CalibratedClassifierCV model trained on clinical symptom presentations.
- `triage_vectorizer.joblib`: Scikit-learn TF-IDF Vectorizer with unigrams and bigrams.
- `xray_cnn.keras`: TensorFlow / Keras MobileNetV2 transfer learning model with multi-label sigmoid classification head for 6 thoracic conditions.

## Evaluation Reports
- `triage_evaluation.json`: Measured held-out test split metrics, accuracy, sensitivity, specificity, and confusion matrix for NLP triage.
- `xray_evaluation.json`: Measured held-out test split metrics, per-pathology ROC-AUC, sensitivity, specificity, and F1 scores for the CNN.
- `system_evaluation_report.json`: Consolidated multi-modal validation report including laboratory boundary verification and risk safety compliance.
