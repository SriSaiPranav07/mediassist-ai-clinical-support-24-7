# MediAssist Comprehensive Model Evaluation Report

## 1. Executive Summary
This report presents the empirical evaluation of all trained models and clinical decision rules implemented in **MediAssist v1.0.0**. Evaluations were conducted on held-out test splits without data leakage.

---

## 2. Module A — NLP Symptom Triage Classifier

### Test Configuration
- **Dataset:** Curated clinical presentation corpus (60 samples across Emergency, Urgent, and Routine presentations).
- **Split:** 75% stratified training, 25% held-out test split.
- **Model:** CalibratedClassifierCV (Platt scaling with balanced Logistic Regression).

### Performance Metrics
| Metric | Measured Value |
| :--- | :--- |
| **Accuracy** | 66.67% |
| **Macro Sensitivity (Recall)** | 66.67% |
| **Macro Specificity** | 83.33% |
| **Macro Precision** | 66.67% |
| **Macro F1-Score** | 66.67% |
| **Macro ROC-AUC** | 0.7733 |

### Confusion Matrix
| True \ Predicted | Emergency Warning | Urgent Assessment | Routine Assessment |
| :--- | :---: | :---: | :---: |
| **Emergency Warning** | 4 | 0 | 1 |
| **Urgent Assessment** | 1 | 3 | 1 |
| **Routine Assessment** | 0 | 2 | 3 |

### Safety Trade-Off Analysis
In clinical triage, false negatives for `Emergency Warning` are potentially catastrophic. To eliminate algorithmic false negatives for acute life threats, MediAssist implements a deterministic clinical red-flag filter before model inference. This guarantees that explicit red flags (crushing chest pain, FAST stroke signs, anaphylaxis, severe hypoxemia) are never downgraded.

---

## 3. Module B — Chest X-Ray Screening CNN (MobileNetV2)

### Test Configuration
- **Architecture:** MobileNetV2 Transfer Learning with Sigmoid multi-label classification head.
- **Loss:** Binary Crossentropy with Adam optimizer (lr=1e-3).
- **Evaluation Threshold:** 0.40.

### Per-Pathology Test Metrics
| Thoracic Pathology | Sensitivity (Recall) | Specificity | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Atelectasis** | 100.0% | 66.7% | 0.944 | 0.600 |
| **Cardiomegaly** | 100.0% | 100.0% | 1.000 | 1.000 |
| **Effusion** | 66.7% | 91.7% | 0.944 | 0.667 |
| **Infiltration** | 100.0% | 46.2% | 1.000 | 0.364 |
| **Pneumothorax** | 100.0% | 69.2% | 0.731 | 0.500 |
| **Normal** | 100.0% | 81.8% | 0.909 | 0.800 |

---

## 4. Module C — Clinical Laboratory Interpreter

### Test Configuration
- **Boundary Verification:** 26 boundary scenarios tested across normal, critical-low, critical-high, sex-differentiated, and fasting-dependent intervals.
- **Pass Rate:** 26 / 26 (100.0% compliance).

---

## 5. Module D — Composite Risk Prioritization Engine

### Test Configuration
- Evaluated against 5 diverse demonstration patient cases (STEMI, Community-Acquired Pneumonia, Upper Respiratory Tract Infection, Acute Ischemic Stroke, Diabetic Ketoacidosis).
- **Emergency Safety Compliance:** 100.0% (Zero downgrades of emergency life threats).
