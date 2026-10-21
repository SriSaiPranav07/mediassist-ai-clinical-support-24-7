# Model Card: Chest X-Ray Screening CNN (MobileNetV2)

## Model Overview
- **Model Name:** MediAssist MobileNetV2 Chest X-Ray Screener
- **Model Version:** 1.0.0
- **Architecture:** MobileNetV2 Backbone (pre-trained on ImageNet) + Custom Multi-Label Head (GlobalAveragePooling2D, BatchNormalization, Dropout 0.3, Dense 128 ReLU, Dropout 0.2, Dense 6 Sigmoid)
- **Target Pathologies:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Pneumothorax, Normal
- **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the `out_relu` feature map

## Intended Use
Assists in preliminary radiological screening by computing multi-label probabilities for 6 common thoracic conditions and generating visual attention saliency maps.

## Prohibited Uses
- Standalone diagnostic radiologic report generation.
- Surgical or interventional planning.

## Empirical Performance (Held-out Test Split)
- **Cardiomegaly:** Sensitivity 100.0%, Specificity 100.0%, ROC-AUC 1.000, F1 1.000
- **Effusion:** Sensitivity 66.7%, Specificity 91.7%, ROC-AUC 0.944, F1 0.667
- **Infiltration:** Sensitivity 100.0%, Specificity 46.2%, ROC-AUC 1.000, F1 0.364
- **Atelectasis:** Sensitivity 100.0%, Specificity 66.7%, ROC-AUC 0.944, F1 0.600
- **Pneumothorax:** Sensitivity 100.0%, Specificity 69.2%, ROC-AUC 0.731, F1 0.500
- **Normal:** Sensitivity 100.0%, Specificity 81.8%, ROC-AUC 0.909, F1 0.800

## Explainability & Saliency
Grad-CAM heatmaps highlight regions of high gradient flow in the final convolutional layer. They are an interpretability tool indicating where the model focused, not a histological segmentation mask.
