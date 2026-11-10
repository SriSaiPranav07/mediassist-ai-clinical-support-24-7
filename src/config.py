"""
MediAssist - Configuration & Clinical Reference Knowledge Base
Author: MediAssist Clinical AI Team
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLE_XRAYS_DIR = DATA_DIR / "sample_xrays"

TRIAGE_MODEL_PATH = MODELS_DIR / "triage_classifier.joblib"
TRIAGE_VECTORIZER_PATH = MODELS_DIR / "triage_vectorizer.joblib"
XRAY_MODEL_PATH = MODELS_DIR / "xray_cnn.keras"

# Triage Categories
TRIAGE_EMERGENCY = "Emergency Warning"
TRIAGE_URGENT = "Urgent Assessment"
TRIAGE_ROUTINE = "Routine Assessment"
TRIAGE_INSUFFICIENT = "Insufficient Information"

TRIAGE_CLASSES = [
    TRIAGE_EMERGENCY,
    TRIAGE_URGENT,
    TRIAGE_ROUTINE,
]

# Emergency Contact Information
EMERGENCY_CONTACT_INDIA = "112"
EMERGENCY_CONTACT_INTERNATIONAL = "Local Emergency Medical Services (e.g. 112 / 911 / 999)"

# Deterministic Safety Layer: Clinical Red Flags requiring immediate emergency care
# Sourced from Manchester Triage System & Emergency Severity Index (ESI) Tier 1 criteria
RED_FLAG_PATTERNS = [
    r"\b(crushing chest pain|chest pressure|radiating to (left )?arm|radiating to jaw)\b",
    r"\b(severe shortness of breath|cannot breathe|gasping for air|stridor|choking|asphyxia)\b",
    r"\b(facial droop|face drooping|arm weakness|slurred speech|sudden hemiparesis|stroke)\b",
    r"\b(unconscious|unresponsive|collapsed|loss of consciousness|passed out and not waking)\b",
    r"\b(anaphylaxis|swollen tongue|throat closing up|angioedema)\b",
    r"\b(coughing up large amounts of blood|massive hemoptysis|vomiting blood|hematemesis)\b",
    r"\b(sudden thunderclap headache|worst headache of life)\b",
    r"\b(severe uncontrolled bleeding|arterial bleed|hemorrhage)\b",
]

# Chest X-Ray Target Screening Classes (NIH ChestX-ray14 derived subset)
XRAY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Pneumothorax",
    "Normal"
]
XRAY_IMG_SIZE = (224, 224)

# Laboratory Reference Intervals and Critical Boundaries
# Sources:
# 1. Tietz Clinical Guide to Laboratory Tests, 4th Edition
# 2. Harrison's Principles of Internal Medicine, 21st Edition
# 3. College of American Pathologists (CAP) Critical Value Reporting Guidelines
LAB_REFERENCE_RANGES = {
    "Hemoglobin": {
        "unit": "g/dL",
        "description": "Oxygen-carrying metalloprotein in red blood cells",
        "ranges": {
            "Male": (13.8, 17.2),
            "Female": (12.1, 15.1),
            "All": (12.1, 17.2)
        },
        "critical_low": 7.0,
        "critical_high": 20.0,
        "low_implication": "Risk of severe tissue hypoxia, acute anemia, or occult hemorrhage.",
        "high_implication": "Polycythemia vera, chronic severe hypoxemia, or profound hemoconcentration."
    },
    "WBC": {
        "unit": "10^3/uL",
        "description": "White Blood Cell Count (Leukocytes)",
        "ranges": {
            "All": (4.5, 11.0)
        },
        "critical_low": 2.0,
        "critical_high": 30.0,
        "low_implication": "Severe neutropenia / bone marrow suppression (high opportunistic infection risk).",
        "high_implication": "Severe leukocytosis, overwhelming sepsis, or hematologic malignancy."
    },
    "Platelets": {
        "unit": "10^3/uL",
        "description": "Thrombocyte count essential for primary hemostasis",
        "ranges": {
            "All": (150.0, 450.0)
        },
        "critical_low": 20.0,
        "critical_high": 1000.0,
        "low_implication": "Profound thrombocytopenia; elevated risk of spontaneous intracranial or internal bleeding.",
        "high_implication": "Severe thrombocytosis with increased risk of microvascular thrombosis or bleeding."
    },
    "Glucose": {
        "unit": "mg/dL",
        "description": "Serum/Plasma Glucose (Fasting or Random context)",
        "ranges": {
            "Fasting": (70.0, 99.0),
            "Random": (70.0, 140.0),
            "All": (70.0, 140.0)
        },
        "critical_low": 50.0,
        "critical_high": 400.0,
        "low_implication": "Severe neuroglycopenia, impending seizure, coma, or permanent brain injury.",
        "high_implication": "Risk of Diabetic Ketoacidosis (DKA) or Hyperosmolar Hyperglycemic State (HHS)."
    },
    "Creatinine": {
        "unit": "mg/dL",
        "description": "End-product of muscle catabolism; marker of renal filtration",
        "ranges": {
            "Male": (0.7, 1.3),
            "Female": (0.6, 1.1),
            "All": (0.6, 1.3)
        },
        "critical_low": None,
        "critical_high": 4.0,
        "low_implication": "Reduced muscle mass, cachexia, or pregnancy (rarely of acute danger).",
        "high_implication": "Acute Kidney Injury (AKI) stage 3 or acute-on-chronic uremic toxicity."
    },
    "Potassium": {
        "unit": "mEq/L",
        "description": "Major intracellular cation; governs cardiac membrane resting potential",
        "ranges": {
            "All": (3.5, 5.0)
        },
        "critical_low": 2.8,
        "critical_high": 6.2,
        "low_implication": "Severe hypokalemia: ventricular ectopy, Torsades de Pointes, respiratory paralysis.",
        "high_implication": "Severe hyperkalemia: peaked T waves, conduction block, ventricular fibrillation, asystole."
    },
    "Sodium": {
        "unit": "mEq/L",
        "description": "Major extracellular cation; governs serum osmolality and fluid shifts",
        "ranges": {
            "All": (135.0, 145.0)
        },
        "critical_low": 120.0,
        "critical_high": 160.0,
        "low_implication": "Severe hyponatremia: cerebral edema, seizures, brain herniation, coma.",
        "high_implication": "Severe hypernatremia: intracellular dehydration, subdural hematoma, delirium."
    },
    "ALT": {
        "unit": "U/L",
        "description": "Alanine Aminotransferase; primary marker of acute hepatocellular injury",
        "ranges": {
            "Male": (10.0, 50.0),
            "Female": (7.0, 35.0),
            "All": (7.0, 50.0)
        },
        "critical_low": None,
        "critical_high": 1000.0,
        "low_implication": "Physiologically normal or minimal clinical significance.",
        "high_implication": "Hyperacute liver injury (e.g., toxic acetaminophen, acute viral hepatitis, ischemic hepatitis)."
    },
    "AST": {
        "unit": "U/L",
        "description": "Aspartate Aminotransferase; indicator of hepatic and musculoskeletal damage",
        "ranges": {
            "Male": (10.0, 40.0),
            "Female": (9.0, 32.0),
            "All": (9.0, 40.0)
        },
        "critical_low": None,
        "critical_high": 1000.0,
        "low_implication": "Normal clinical variation.",
        "high_implication": "Severe acute hepatic necrosis, profound rhabdomyolysis, or severe myocardial injury."
    },
    "Troponin_I": {
        "unit": "ng/mL",
        "description": "Cardiac-specific biomarker for myocardial infarction",
        "ranges": {
            "All": (0.0, 0.04)
        },
        "critical_low": None,
        "critical_high": 0.40,
        "low_implication": "Within normal baseline.",
        "high_implication": "Myocardial necrosis consistent with Acute Coronary Syndrome (STEMI / NSTEMI)."
    }
}

# Vital Signs Reference Intervals
VITALS_REFERENCE = {
    "HeartRate": {"unit": "bpm", "normal_min": 60, "normal_max": 100, "critical_low": 40, "critical_high": 140},
    "SpO2": {"unit": "%", "normal_min": 95, "normal_max": 100, "critical_low": 90, "critical_high": 101},
    "SystolicBP": {"unit": "mmHg", "normal_min": 90, "normal_max": 130, "critical_low": 80, "critical_high": 180},
    "DiastolicBP": {"unit": "mmHg", "normal_min": 60, "normal_max": 85, "critical_low": 50, "critical_high": 110},
    "RespiratoryRate": {"unit": "breaths/min", "normal_min": 12, "normal_max": 20, "critical_low": 8, "critical_high": 30},
    "Temperature": {"unit": "deg C", "normal_min": 36.5, "normal_max": 37.5, "critical_low": 35.0, "critical_high": 39.5}
}

# Patient Risk Priority Levels
RISK_LEVEL_EMERGENCY = "Emergency Review"
RISK_LEVEL_HIGH = "High Priority"
RISK_LEVEL_STANDARD = "Standard Review"
RISK_LEVEL_INSUFFICIENT = "Insufficient Information"
