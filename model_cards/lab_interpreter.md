# Model Card: Clinical Laboratory Interpreter & Critical Value Engine

## Model Details
- **Module Name:** MediAssist Laboratory Result Interpreter
- **Version:** 1.0.0
- **Type:** Expert Deterministic Rule Engine
- **Knowledge Sources:**
  - Tietz Clinical Guide to Laboratory Tests, 4th Edition
  - Harrison's Principles of Internal Medicine, 21st Edition
  - College of American Pathologists (CAP) Critical Value Standards

## Supported Laboratory Panels & Parameters
1. **Hemoglobin (g/dL):** Sex-specific reference ranges (Male: 13.8-17.2, Female: 12.1-15.1); Critical limits: < 7.0 and > 20.0 g/dL.
2. **White Blood Cell (WBC) Count (10^3/uL):** Reference 4.5 - 11.0; Critical limits: < 2.0 and > 30.0.
3. **Platelet Count (10^3/uL):** Reference 150.0 - 450.0; Critical limits: < 20.0 and > 1000.0.
4. **Serum Glucose (mg/dL):** Context-specific (Fasting: 70-99, Random: 70-140); Critical limits: < 50.0 and > 400.0.
5. **Serum Creatinine (mg/dL):** Sex-specific (Male: 0.7-1.3, Female: 0.6-1.1); Critical limit: > 4.0.
6. **Serum Potassium (mEq/L):** Reference 3.5 - 5.0; Critical limits: < 2.8 and > 6.2.
7. **Serum Sodium (mEq/L):** Reference 135.0 - 145.0; Critical limits: < 120.0 and > 160.0.
8. **ALT & AST (U/L):** Hepatic injury markers; Critical limit: > 1000.0 U/L.
9. **Troponin I (ng/mL):** Cardiac necrosis marker; Critical limit: > 0.40 ng/mL.

## Safety Guarantees
- Any critical value trigger automatically escalates the patient to Emergency Review priority.
- Unknown tests or unit mismatches are rejected with explicit guidance rather than guessing or silently misconverting.
