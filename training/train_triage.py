"""
MediAssist - Symptom Triage NLP Training Pipeline
Author: Zelvora Industry Labs / MediAssist Team
Trains a Calibrated TF-IDF Classifier on clinical symptom presentations.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE,
    TRIAGE_MODEL_PATH,
    TRIAGE_VECTORIZER_PATH,
    MODELS_DIR,
    DATA_DIR
)
from src.evaluation import calculate_classification_metrics

# Curated Clinical Triage Presentations across three urgency categories
CLINICAL_TRIAGE_RECORDS = [
    # EMERGENCY WARNING
    ("Crushing substernal chest pain radiating to left arm and jaw with profuse cold sweating", TRIAGE_EMERGENCY),
    ("Severe pressure in center of chest with shortness of breath and dizziness", TRIAGE_EMERGENCY),
    ("Sudden onset of right-sided arm and leg weakness with facial drooping and slurred speech", TRIAGE_EMERGENCY),
    ("Patient is unresponsive, collapsed on the floor, breathing shallowly", TRIAGE_EMERGENCY),
    ("Severe anaphylaxis with swollen tongue, throat closing up, and audible stridor", TRIAGE_EMERGENCY),
    ("Coughing up large quantities of bright red blood with severe respiratory distress", TRIAGE_EMERGENCY),
    ("Sudden thunderclap headache, worst pain of life, followed by confusion and stiff neck", TRIAGE_EMERGENCY),
    ("Massive hematemesis with dizziness, pale clammy skin, and tachycardia", TRIAGE_EMERGENCY),
    ("Severe shortness of breath, cannot speak in full sentences, gasping for air", TRIAGE_EMERGENCY),
    ("Arterial bleeding from deep forearm laceration not stopping with direct pressure", TRIAGE_EMERGENCY),
    ("Severe central chest tightness with radiating pain to neck and impending sense of doom", TRIAGE_EMERGENCY),
    ("Sudden loss of vision in right eye with severe facial weakness and hemiparesis", TRIAGE_EMERGENCY),
    ("Infant presenting with limpness, cyanosis around lips, and difficulty breathing", TRIAGE_EMERGENCY),
    ("Stabbing retrosternal chest pain with syncope and cold clammy diaphoresis", TRIAGE_EMERGENCY),
    ("Acute acute dyspnea, choking sensation, and inability to swallow saliva", TRIAGE_EMERGENCY),
    ("Sudden complete weakness in left extremities and inability to speak coherently", TRIAGE_EMERGENCY),
    ("Massive gastrointestinal bleeding with syncope and profound hypotension", TRIAGE_EMERGENCY),
    ("Severe burning chest pain radiating to left shoulder with shortness of breath and nausea", TRIAGE_EMERGENCY),
    ("Sudden respiratory failure with SpO2 dropping to 82% and cyanosis", TRIAGE_EMERGENCY),
    ("Unconscious adult found after cardiac arrest with agonal breathing", TRIAGE_EMERGENCY),

    # URGENT ASSESSMENT
    ("High fever of 39 C for 3 days with productive cough, yellow phlegm, and localized pleuritic pain", TRIAGE_URGENT),
    ("Persistent severe lower right abdominal pain worsening over 12 hours with nausea and vomiting", TRIAGE_URGENT),
    ("Acute painful swollen red left calf after long flight with mild shortness of breath", TRIAGE_URGENT),
    ("Asthma exacerbation with wheezing and tightness not relieved by albuterol inhaler", TRIAGE_URGENT),
    ("Severe burning sensation upon urination with flank pain, chills, and high fever", TRIAGE_URGENT),
    ("Deep laceration on hand requiring sutures, bleeding controlled with pressure dressing", TRIAGE_URGENT),
    ("Persistent severe migraine for 48 hours with intractable vomiting and photophobia", TRIAGE_URGENT),
    ("Diabetic patient with high blood sugar 380 mg/dL, extreme thirst, weakness, and fruity breath", TRIAGE_URGENT),
    ("Elderly patient with 2 days of progressive confusion, lethargy, and decreased oral intake", TRIAGE_URGENT),
    ("Acute severe back pain with urinary retention and bilateral lower extremity paresthesia", TRIAGE_URGENT),
    ("Child with persistent high fever 39.5 C, persistent barking cough, and mild stridor", TRIAGE_URGENT),
    ("Worsening shortness of breath on exertion with bilateral ankle edema and orthopnea", TRIAGE_URGENT),
    ("Severe pain and inability to bear weight on deformed swollen right ankle after fall", TRIAGE_URGENT),
    ("Acute abdominal distension and persistent vomiting with inability to pass flatus or stool", TRIAGE_URGENT),
    ("Worsening cellulitis on lower leg with expanding erythema, heat, and red streaking", TRIAGE_URGENT),
    ("Severe dizziness and vertigo with persistent vomiting preventing standing or walking", TRIAGE_URGENT),
    ("Persistent diarrhea and vomiting for 3 days with signs of moderate dehydration and sunken eyes", TRIAGE_URGENT),
    ("Palpitations lasting 2 hours with lightheadedness and pulse rate around 130 bpm", TRIAGE_URGENT),
    ("Severe localized right upper quadrant abdominal pain triggered after fatty meal", TRIAGE_URGENT),
    ("High fever, neck soreness, and body aches in returning international traveler", TRIAGE_URGENT),

    # ROUTINE ASSESSMENT
    ("Mild dry cough and runny nose for 4 days with occasional sneezing and low grade fever 37.4 C", TRIAGE_ROUTINE),
    ("Sore throat and itchy watery eyes with clear nasal discharge consistent with seasonal allergy", TRIAGE_ROUTINE),
    ("Chronic lower back dull ache for 6 months, worse after prolonged sitting, no radiation", TRIAGE_ROUTINE),
    ("Mild intermittent tension headache across forehead after long hours on computer screen", TRIAGE_ROUTINE),
    ("Minor superficial scrape on elbow after trip on sidewalk, cleaned with antiseptic", TRIAGE_ROUTINE),
    ("Gradual knee stiffness in the morning lasting 15 minutes, improving with gentle walking", TRIAGE_ROUTINE),
    ("Routine prescription refill request for stable hypertension medication amlodipine", TRIAGE_ROUTINE),
    ("Mild indigestion and bloating after spicy dinner yesterday, no vomiting or severe pain", TRIAGE_ROUTINE),
    ("Dry itchy skin patch on forearm present for 2 weeks with mild flaking", TRIAGE_ROUTINE),
    ("Follow-up appointment for routine annual wellness physical examination and blood tests", TRIAGE_ROUTINE),
    ("Mild muscle soreness in quadriceps after gym workout two days ago, no swelling", TRIAGE_ROUTINE),
    ("Superficial paper cut on index finger with minimal bleeding that stopped immediately", TRIAGE_ROUTINE),
    ("Mild nasal congestion and sinus pressure without fever, facial swelling, or purulent drainage", TRIAGE_ROUTINE),
    ("Routine monitoring check of fasting blood glucose for well-controlled type 2 diabetes", TRIAGE_ROUTINE),
    ("Mild insomnia and difficulty falling asleep during exam week with stress", TRIAGE_ROUTINE),
    ("Request for seasonal influenza vaccination and travel vaccine consultation", TRIAGE_ROUTINE),
    ("Mild occasional heartburn relieved by over-the-counter antacid tablets", TRIAGE_ROUTINE),
    ("Small painless benign-appearing skin tag on neck requesting cosmetic removal", TRIAGE_ROUTINE),
    ("Mild dandruff and dry scalp itching without hair loss or skin lesions", TRIAGE_ROUTINE),
    ("Routine check of blood pressure, currently asymptomatic and feeling healthy", TRIAGE_ROUTINE)
]

def train_and_evaluate_triage():
    print("=" * 60)
    print("MediAssist: Training NLP Symptom Triage Model")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(CLINICAL_TRIAGE_RECORDS, columns=["text", "urgency_category"])
    
    # Save training dataset to CSV for reproducibility & audit
    csv_path = DATA_DIR / "triage_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path} (Total samples: {len(df)})")
    print("Class Distribution:")
    print(df["urgency_category"].value_counts())

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["urgency_category"],
        test_size=0.25,
        random_state=42,
        stratify=df["urgency_category"]
    )

    # Feature extraction: TF-IDF with unigrams and bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=250,
        sublinear_tf=True,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Base Logistic Regression with balanced class weights
    base_clf = LogisticRegression(
        C=1.5,
        class_weight="balanced",
        max_iter=500,
        random_state=42
    )

    # Calibrated Classifier for accurate clinical probabilities
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
    calibrated_clf.fit(X_train_vec, y_train)

    # Evaluation on held-out test split
    y_pred = calibrated_clf.predict(X_test_vec)
    y_prob = calibrated_clf.predict_proba(X_test_vec)

    class_names = list(calibrated_clf.classes_)
    metrics = calculate_classification_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names
    )

    print("\nHeld-out Test Evaluation Results:")
    print(f"Accuracy:          {metrics['Accuracy'] * 100:.2f}%")
    print(f"Macro Sensitivity: {metrics['Sensitivity_Macro'] * 100:.2f}%")
    print(f"Macro Specificity: {metrics['Specificity_Macro'] * 100:.2f}%")
    print(f"Macro Precision:   {metrics['Precision_Macro'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics['F1_Macro'] * 100:.2f}%")
    print(f"Macro ROC-AUC:     {metrics['ROC_AUC_Macro']}")

    # Save model artifacts
    joblib.dump(calibrated_clf, TRIAGE_MODEL_PATH)
    joblib.dump(vectorizer, TRIAGE_VECTORIZER_PATH)
    print(f"\nModel saved to: {TRIAGE_MODEL_PATH}")
    print(f"Vectorizer saved to: {TRIAGE_VECTORIZER_PATH}")

    # Save metrics JSON
    metrics_path = MODELS_DIR / "triage_evaluation.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report written to: {metrics_path}")

    return metrics

if __name__ == "__main__":
    train_and_evaluate_triage()
