"""
MediAssist Models Module
Encapsulates model loading and inference logic with Streamlit caching.
Author: MediAssist Clinical AI Team
"""
from typing import Optional
import streamlit as st

from .triage_model import SymptomTriageModel
from .xray_model import ChestXrayModel
from .lab_model import LabInterpreterModel
from .risk_engine import CompositeRiskEngine
from .rag_engine import MedicalGuidelineRAGEngine


@st.cache_resource(show_spinner=False)
def get_triage_model() -> SymptomTriageModel:
    """Cached singleton loader for NLP Symptom Triage Model."""
    return SymptomTriageModel()


@st.cache_resource(show_spinner=False)
def get_xray_model() -> ChestXrayModel:
    """Cached singleton loader for MobileNetV2 Chest X-Ray Screener."""
    return ChestXrayModel()


@st.cache_resource(show_spinner=False)
def get_lab_model() -> LabInterpreterModel:
    """Cached singleton loader for Laboratory Interpreter Engine."""
    return LabInterpreterModel()


@st.cache_resource(show_spinner=False)
def get_risk_engine() -> CompositeRiskEngine:
    """Cached singleton loader for Multi-Modal Risk Scoring Engine."""
    return CompositeRiskEngine()


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> MedicalGuidelineRAGEngine:
    """Cached singleton loader for Medical Guideline RAG Engine."""
    return MedicalGuidelineRAGEngine()


__all__ = [
    "SymptomTriageModel",
    "ChestXrayModel",
    "LabInterpreterModel",
    "CompositeRiskEngine",
    "MedicalGuidelineRAGEngine",
    "get_triage_model",
    "get_xray_model",
    "get_lab_model",
    "get_risk_engine",
    "get_rag_engine"
]
