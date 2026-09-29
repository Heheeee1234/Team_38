from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0.0-p4"

@dataclass(frozen=True)
class PatientContext:
    age: int | float | None = None
    sex: str | None = None
    relevant_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    recent_labs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TrendEvidence:
    score: float = 0.0
    direction: str = "stable"
    persistent_deterioration: bool = False
    contributing_signals: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AnomalyEvidence:
    probability: float = 0.0
    is_anomalous: bool = False
    contributing_features: list[str] = field(default_factory=list)
    model_version: int | None = None

@dataclass(frozen=True)
class RiskEvidence:
    risk_score: float = 0.0
    risk_level: str = "LOW"
    contributing_factors: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class SimilarCase:
    case_id: str
    similarity: float
    physiological_pattern: dict[str, Any]
    anomaly_confidence: float
    detected_trends: dict[str, Any]
    clinician_decision: str | None
    clinician_feedback: str | None
    outcome: str | None
    experience_confidence: float = 0.0

@dataclass(frozen=True)
class ConsolidatedEvidence:
    """P3 -> P4 boundary. P4 never recomputes risk/anomaly/trend."""
    schema_version: str
    evidence_id: str
    patient_id: str
    window_start: str | None
    window_end: str
    current_vitals: dict[str, float]
    trend: TrendEvidence
    anomaly: AnomalyEvidence
    risk: RiskEvidence
    patient_context: PatientContext
    experience_cases: list[SimilarCase] = field(default_factory=list)
    experience_confidence: float = 0.0
    fusion_confidence: float = 0.0
    conflicts_or_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class Citation:
    document_id: str
    version: str
    locator: str
    source_url: str = ""
    score: float = 0.0
    chunk_text: str = ""

@dataclass(frozen=True)
class ReasoningOutput:
    schema_version: str
    patient_id: str
    evidence_id: str
    generated_at: str
    explanation: str
    contributing_factors: list[str]
    recommended_action: str
    citations: list[Citation]
    similar_cases_referenced: list[str]
    confidence: float
    model: str
    tool_call_count: int = 0
    disclaimer: str = "Decision support only; not a diagnosis or treatment order."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
