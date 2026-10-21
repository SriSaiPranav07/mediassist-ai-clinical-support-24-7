"""
MediAssist - Explainability Center (LIME, SHAP, and Clinician Explanations)
Author: Zelvora Industry Labs / MediAssist Team
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

class ExplainabilityEngine:
    """
    Coordinates interpretability methods across NLP, Tabular/Clinical, and Imaging models.
    """

    @staticmethod
    def format_10_second_explanation(finding: str, reason: str, confidence: str, limitation: str) -> str:
        """
        Formats clinician-ready 10-second summary according to capstone specification.
        """
        return (
            f"**Finding:** {finding}\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Confidence:** {confidence}\n\n"
            f"**Limitation:** {limitation}"
        )

    @staticmethod
    def explain_text_with_lime(text: str, classifier, vectorizer, num_features: int = 6) -> Optional[Dict[str, Any]]:
        """
        Generates Local Interpretable Model-agnostic Explanations (LIME) for symptom triage text.
        """
        try:
            from lime.lime_text import LimeTextExplainer

            class_names = list(classifier.classes_)
            def predict_fn(texts):
                feats = vectorizer.transform(texts)
                return classifier.predict_proba(feats)

            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                labels=[0, 1, 2] if len(class_names) >= 3 else [0, 1]
            )

            # Extract weights for the predicted top class
            top_class_idx = int(np.argmax(predict_fn([text])[0]))
            predicted_label = class_names[top_class_idx]
            weights = exp.as_list(label=top_class_idx)

            return {
                "available": True,
                "predicted_label": predicted_label,
                "weights": weights,  # List of (word, score)
                "as_html": exp.as_html()
            }
        except Exception as e:
            # Fallback to feature importance from vectorizer
            return ExplainabilityEngine.fallback_word_importance(text, classifier, vectorizer, top_n=num_features)

    @staticmethod
    def fallback_word_importance(text: str, classifier, vectorizer, top_n: int = 6) -> Dict[str, Any]:
        """
        Deterministic TF-IDF token weight fallback when LIME environment is constrained.
        """
        try:
            tokens = text.lower().split()
            vocab = vectorizer.vocabulary_
            weights = []
            classes = list(classifier.classes_)
            
            # Predict top class
            feat = vectorizer.transform([text])
            top_class_idx = int(np.argmax(classifier.predict_proba(feat)[0]))
            predicted_label = classes[top_class_idx]

            coef = classifier.coef_[top_class_idx] if hasattr(classifier, "coef_") else None
            for token in set(tokens):
                cleaned = "".join(c for c in token if c.isalnum())
                if cleaned in vocab and coef is not None:
                    idx = vocab[cleaned]
                    score = float(coef[idx])
                    weights.append((cleaned, score))

            weights = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:top_n]
            return {
                "available": True,
                "predicted_label": predicted_label,
                "weights": weights,
                "as_html": None
            }
        except Exception:
            return {"available": False, "weights": [], "predicted_label": "Unknown", "as_html": None}

    @staticmethod
    def explain_tabular_risk_factors(
        sub_scores: Dict[str, float],
        contributing_factors: List[str]
    ) -> pd.DataFrame:
        """
        Builds a structured feature attribution dataframe representing risk contribution.
        """
        data = []
        for category, score in sub_scores.items():
            data.append({
                "Clinical Dimension": category.capitalize(),
                "Risk Contribution Points": score,
                "Percentage of Max": round((score / (30.0 if category == "symptoms" else 25.0 if category in ["vitals", "labs"] else 20.0)) * 100, 1)
            })
        return pd.DataFrame(data)

    @staticmethod
    def generate_shap_waterfall_data(sub_scores: Dict[str, float], base_score: float = 10.0) -> Dict[str, Any]:
        """
        Generates waterfall values suitable for Plotly SHAP-like attribution charts.
        """
        dimensions = ["Base Risk", "Symptoms", "Vitals", "Laboratory", "Radiology"]
        values = [
            base_score,
            sub_scores.get("symptoms", 0.0),
            sub_scores.get("vitals", 0.0),
            sub_scores.get("labs", 0.0),
            sub_scores.get("radiology", 0.0)
        ]
        measures = ["absolute", "relative", "relative", "relative", "relative"]
        return {
            "dimensions": dimensions,
            "values": values,
            "measures": measures
        }
