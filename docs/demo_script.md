# MediAssist — Three-Minute Video Demonstration Script

**Title:** MediAssist: Multi-Model AI Clinical Decision Support System  
**Presenter:** AI/ML Capstone Engineer  
**Duration:** 3:00 Minutes  
**Target Audience:** Clinical Mentors, Healthcare Researchers & Evaluation Panel  

---

### Segment 1: Project Overview & Problem Statement [0:00 - 0:30]
- **Visual:** Display Streamlit landing page with the MediAssist banner and the safety disclaimer.
- **Narrator Speech:**  
  *"Hello everyone. Welcome to MediAssist — a multi-model AI clinical decision support system built for the Zelvora Industry Labs AI Capstone. Hospital emergency departments and triage units face severe diagnostic bottlenecks. MediAssist demonstrates how multiple specialized AI models — natural language processing, deep computer vision, and expert laboratory rules — can synthesize clinical data into actionable, explainable insights while enforcing strict patient safety overrides. Importantly, MediAssist is an educational research demonstration, not a standalone diagnostic device."*

---

### Segment 2: NLP Symptom Triage & 10-Second Explanations [0:30 - 1:00]
- **Visual:** Navigate to `2_Symptom_Triage`. Select demonstration case `PT-101 (Suspected STEMI)`.
- **Narrator Speech:**  
  *"Let's examine patient PT-101 presenting with acute crushing chest pain radiating to the left arm. When we click 'Run Triage Screening', our deterministic safety layer immediately detects this acute life-threat, flagging an 'Emergency Warning' with 100% priority and advising immediate contact with emergency services — dialing 112 in India. Underneath, our mandatory capstone mentor challenge is met: every decision produces a plain-English explanation digestible in under 10 seconds: structured with Finding, Reason, Confidence, and Limitation."*

---

### Segment 3: Chest X-Ray Screening & Grad-CAM Saliency [1:00 - 1:30]
- **Visual:** Navigate to `3_Chest_Xray_Analysis`. Select `cardiomegaly_sample.png`.
- **Narrator Speech:**  
  *"Next, we move to Module B: chest radiograph screening. We pass the radiograph through our fine-tuned MobileNetV2 CNN. The model screens for 6 target pathologies simultaneously using multi-label sigmoid heads. Here, it identifies elevated probability for Cardiomegaly. Alongside the probability, MediAssist renders a Grad-CAM saliency map overlaying model attention directly onto the thoracic cardiac silhouette, allowing the clinician to immediately verify where the network focused."*

---

### Segment 4: Laboratory Interpretation & Critical Alerts [1:30 - 2:00]
- **Visual:** Navigate to `4_Lab_Interpreter`. Pre-fill `PT-101` or enter Potassium = 6.8 mEq/L and Troponin I = 1.45 ng/mL.
- **Narrator Speech:**  
  *"In Module C, we evaluate laboratory panels using authoritative reference intervals from the Tietz and Harrison compendiums. When we input elevated Troponin I and Potassium, the engine flags these as 'Critical High'. These critical limits trigger emergency protocol alerts that cannot be muted or ignored by downstream models."*

---

### Segment 5: Transparent Patient Risk Prioritization [2:00 - 2:30]
- **Visual:** Navigate to `5_Patient_Risk_Assessment`. Select `PT-101`.
- **Narrator Speech:**  
  *"Module D synthesizes all modalities into a transparent patient prioritization index. Notice how our safety guarantee works: because an emergency warning was detected in the symptom and lab modules, the patient is assigned 'Emergency Review' with a floor score of 85. Our architecture strictly forbids diluting life-threatening red flags by averaging them with normal scores elsewhere."*

---

### Segment 6: Explainability, Evaluation & Safe Edge Cases [2:30 - 3:00]
- **Visual:** Briefly show `6_Explainability_Center` (LIME tokens), `7_Model_Evaluation` (Confusion matrix), and an invalid input edge case (submitting empty text or a non-image file).
- **Narrator Speech:**  
  *"Finally, our Explainability Center features LIME token attributions, and our Evaluation page presents actual measured test-set ROC-AUC and sensitivity metrics — zero fake numbers. When invalid text or corrupted images are submitted, MediAssist safely rejects the input without crashing. In conclusion, MediAssist balances machine learning power with deterministic clinical safety, human-in-the-loop oversight, and 10-second explainability. Thank you."*
