# Model Card: NLP Symptom Triage Classifier

## Model Overview
- **Model Name:** MediAssist Calibrated Symptom Triage Classifier
- **Model Version:** 1.0.0
- **Type:** Calibrated Linear Classifier (Platt Sigmoid Scaling over Balanced Logistic Regression)
- **Feature Extractor:** TF-IDF Vectorizer (unigrams + bigrams, sublinear TF scaling, max 250 features)

## Intended Use
Provides preliminary prioritization of patient symptom narratives into three urgency levels:
- `Emergency Warning`
- `Urgent Assessment`
- `Routine Assessment`
- `Insufficient Information` (for invalid or too-short inputs)

## Safety Layer & Fallback
Includes an explicit deterministic regex-based clinical red-flag filter before statistical inference. If severe chest pain, stroke symptoms (FAST), acute airway compromise, or massive bleeding is detected, the safety filter immediately flags `Emergency Warning` with 100% priority, advising immediate dispatch/referral to 112 (India).

## Empirical Performance (Held-out Test Split)
- **Accuracy:** 66.7%
- **Macro Sensitivity (Recall):** 66.7%
- **Macro Specificity:** 83.3%
- **Macro Precision:** 66.7%
- **Macro F1-Score:** 66.7%
- **Macro ROC-AUC:** 0.773

## Limitations & Bias
- Trained on English symptom narratives. May not perform well on regional slang, multilingual colloquialisms, or ambiguous descriptions.
- Requires clinical verification by a registered triage nurse or physician.
