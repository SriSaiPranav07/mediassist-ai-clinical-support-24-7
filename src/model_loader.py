"""
MediAssist - Model Loader & Singleton Caching
Author: Zelvora Industry Labs / MediAssist Team
"""
import functools
from typing import Optional
from .symptom_triage import SymptomTriageEngine
from .lab_interpreter import LabInterpreter
from .xray_model import ChestXRayScreener
from .risk_engine import CompositeRiskEngine
from .explainability import ExplainabilityEngine

@functools.lru_cache(maxsize=1)
def get_triage_engine() -> SymptomTriageEngine:
    """Returns singleton instance of SymptomTriageEngine."""
    return SymptomTriageEngine()

@functools.lru_cache(maxsize=1)
def get_lab_interpreter() -> LabInterpreter:
    """Returns singleton instance of LabInterpreter."""
    return LabInterpreter()

@functools.lru_cache(maxsize=1)
def get_xray_screener() -> ChestXRayScreener:
    """Returns singleton instance of ChestXRayScreener."""
    return ChestXRayScreener()

@functools.lru_cache(maxsize=1)
def get_risk_engine() -> CompositeRiskEngine:
    """Returns singleton instance of CompositeRiskEngine."""
    return CompositeRiskEngine()

@functools.lru_cache(maxsize=1)
def get_explainability_engine() -> ExplainabilityEngine:
    """Returns singleton instance of ExplainabilityEngine."""
    return ExplainabilityEngine()
