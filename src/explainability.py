"""
MediAssist - Explainability Center (LIME, SHAP, and AI Clinical Decision Support Summaries)
Author: MediAssist Clinical AI Team
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class ExplainabilityEngine:
    """
    Coordinates interpretability methods across NLP, Tabular/Clinical, and Imaging models.
    """

    @staticmethod
    def format_clinical_summary(
        finding: str,
        reason: str,
        confidence: str,
        limitation: str,
        missing_info: Optional[List[str]] = None
    ) -> str:
        """
        Formats structured AI-Generated Clinical Decision Support Summary.
        """
        base = (
            f"**Finding:** {finding}\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Confidence:** {confidence}\n\n"
            f"**Limitation:** {limitation}"
        )
        if missing_info:
            base += f"\n\n**Missing Details / Recommended Next Inquiries:** {'; '.join(missing_info)}"
        return base

    @staticmethod
    def explain_text_with_lime(
        text: str,
        classifier: Any,
        vectorizer: Any,
        target_class: Optional[str] = None,
        num_features: int = 8
    ) -> Dict[str, Any]:
        """
        Generates Local Interpretable Model-agnostic Explanations (LIME) for symptom triage text.
        Explains which tokens drove the classification toward or away from the target class.
        """
        if not text or classifier is None or vectorizer is None:
            return {
                "available": False,
                "predicted_label": "Unknown",
                "weights": [],
                "positive_weights": [],
                "negative_weights": [],
                "disclaimer": "LIME explanation unavailable due to missing text or model assets.",
                "as_html": None
            }

        try:
            from lime.lime_text import LimeTextExplainer

            class_names = list(classifier.classes_)

            def predict_fn(texts):
                feats = vectorizer.transform(texts)
                return classifier.predict_proba(feats)

            # Determine target class index
            probs = predict_fn([text])[0]
            if target_class and target_class in class_names:
                target_idx = class_names.index(target_class)
            else:
                target_idx = int(np.argmax(probs))
            predicted_label = class_names[target_idx]

            explainer = LimeTextExplainer(
                class_names=class_names,
                bow=True,
                split_expression=r"\W+",
                random_state=42
            )
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                labels=[target_idx]
            )

            raw_weights = exp.as_list(label=target_idx)
            pos_weights = [(w, score) for w, score in raw_weights if score > 0]
            neg_weights = [(w, score) for w, score in raw_weights if score < 0]

            return {
                "available": True,
                "predicted_label": predicted_label,
                "target_class": predicted_label,
                "weights": raw_weights,
                "positive_weights": pos_weights,
                "negative_weights": neg_weights,
                "disclaimer": (
                    "LIME highlights local lexical associations within the model's feature space "
                    "for this text input. Token weights reflect statistical model behavior, not medical etiology or biological causation."
                ),
                "as_html": exp.as_html()
            }
        except Exception:
            # Fallback to feature importance from vectorizer
            return ExplainabilityEngine.fallback_word_importance(
                text=text,
                classifier=classifier,
                vectorizer=vectorizer,
                target_class=target_class,
                top_n=num_features
            )

    @staticmethod
    def fallback_word_importance(
        text: str,
        classifier: Any,
        vectorizer: Any,
        target_class: Optional[str] = None,
        top_n: int = 8
    ) -> Dict[str, Any]:
        """
        Deterministic TF-IDF token weight fallback when LIME environment is constrained.
        """
        try:
            tokens = text.lower().split()
            vocab = vectorizer.vocabulary_
            class_names = list(classifier.classes_)

            feat = vectorizer.transform([text])
            probs = classifier.predict_proba(feat)[0]

            if target_class and target_class in class_names:
                target_idx = class_names.index(target_class)
            else:
                target_idx = int(np.argmax(probs))
            predicted_label = class_names[target_idx]

            weights = []
            # Check if underlying estimator has coef_
            coef = None
            if hasattr(classifier, "coef_"):
                coef = classifier.coef_[target_idx]
            elif hasattr(classifier, "calibrated_classifiers_") and len(classifier.calibrated_classifiers_) > 0:
                base_est = classifier.calibrated_classifiers_[0].estimator
                if hasattr(base_est, "coef_"):
                    coef = base_est.coef_[target_idx]

            for token in set(tokens):
                cleaned = "".join(c for c in token if c.isalnum())
                if cleaned in vocab and coef is not None:
                    idx = vocab[cleaned]
                    score = float(coef[idx])
                    weights.append((cleaned, score))
                elif cleaned in vocab:
                    # Pure TF-IDF value fallback
                    col_idx = vocab[cleaned]
                    score = float(feat[0, col_idx])
                    weights.append((cleaned, score))

            weights = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:top_n]
            pos_weights = [(w, score) for w, score in weights if score > 0]
            neg_weights = [(w, score) for w, score in weights if score < 0]

            return {
                "available": True,
                "predicted_label": predicted_label,
                "target_class": predicted_label,
                "weights": weights,
                "positive_weights": pos_weights,
                "negative_weights": neg_weights,
                "disclaimer": (
                    "Lexical alignment scores indicate feature contribution in the model's vocabulary space; "
                    "they describe model classification behavior rather than clinical causation."
                ),
                "as_html": None
            }
        except Exception:
            return {
                "available": False,
                "predicted_label": "Unknown",
                "target_class": "Unknown",
                "weights": [],
                "positive_weights": [],
                "negative_weights": [],
                "disclaimer": "Feature importance calculation unavailable for this input.",
                "as_html": None
            }

    @staticmethod
    def explain_tabular_risk_factors(
        sub_scores: Dict[str, float],
        contributing_factors: List[str]
    ) -> pd.DataFrame:
        """
        Creates structured clinical risk factor explanation table.
        """
        data = []
        for domain, score in sub_scores.items():
            level = "Critical" if score >= 75 else "Elevated" if score >= 40 else "Low"
            matching_factors = [f for f in contributing_factors if domain.lower() in f.lower()]
            data.append({
                "Risk Domain": domain,
                "Domain Score": f"{score:.1f} / 100",
                "Risk Level": level,
                "Active Flags": "; ".join(matching_factors) if matching_factors else "Baseline"
            })
        return pd.DataFrame(data)
