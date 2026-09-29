from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence

from clinical_data_gen.state.patient_state import PatientState
from clinical_data_gen.trend import TrendResult

FEATURE_NAMES: tuple[str, ...] = (
    "HR",
    "O2Sat",
    "Resp",
    "SBP",
    "DBP",
    "trend_score",
    "anomaly_score",
    "risk_score",
)


@dataclass
class HistoricalCase:
    case_id: str
    patient_id: str
    timestamp: Any
    physiological_pattern: dict[str, float]
    trend_information: dict[str, Any]
    anomaly_confidence: float
    risk_score: float
    clinician_decision: str | None = None
    clinician_feedback: str | None = None
    outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "physiological_pattern": dict(self.physiological_pattern),
            "trend_information": dict(self.trend_information),
            "anomaly_confidence": self.anomaly_confidence,
            "risk_score": self.risk_score,
            "clinician_decision": self.clinician_decision,
            "clinician_feedback": self.clinician_feedback,
            "outcome": self.outcome,
            "metadata": dict(self.metadata),
        }


@dataclass
class ExperienceResult:
    retrieved_cases: list[HistoricalCase]
    experience_confidence: float
    similarity_scores: list[float]
    feature_vector: list[float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_cases": [case.to_dict() for case in self.retrieved_cases],
            "experience_confidence": self.experience_confidence,
            "similarity_scores": list(self.similarity_scores),
            "feature_vector": list(self.feature_vector),
            "explanation": self.explanation,
        }


class ExperienceStore:
    """Repository contract for previously observed clinical cases."""

    def save_case(self, case: HistoricalCase) -> HistoricalCase:
        raise NotImplementedError

    def get_case(self, case_id: str) -> HistoricalCase | None:
        raise NotImplementedError

    def search_similar(
        self,
        feature_vector: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[HistoricalCase, float]]:
        raise NotImplementedError

    def update_feedback(
        self,
        case_id: str,
        clinician_decision: str,
        feedback: str,
        outcome: str,
    ) -> HistoricalCase:
        raise NotImplementedError


class InMemoryExperienceStore(ExperienceStore):
    def __init__(self, similarity_scorer: "SimilarityScorer | None" = None) -> None:
        self._cases: dict[str, HistoricalCase] = {}
        self.similarity_scorer = similarity_scorer or CosineSimilarityScorer()

    def save_case(self, case: HistoricalCase) -> HistoricalCase:
        self._cases[case.case_id] = case
        return case

    def get_case(self, case_id: str) -> HistoricalCase | None:
        return self._cases.get(case_id)

    def search_similar(
        self,
        feature_vector: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[HistoricalCase, float]]:
        results: list[tuple[HistoricalCase, float]] = []
        for case in self._cases.values():
            if case.clinician_decision is None and case.outcome is None:
                continue
            candidate_vector = case_feature_vector(case)
            similarity = self.similarity_scorer.similarity(
                list(feature_vector),
                candidate_vector,
            )
            results.append((case, similarity))
        return sorted(results, key=lambda item: item[1], reverse=True)[: max(1, top_k)]

    def update_feedback(
        self,
        case_id: str,
        clinician_decision: str,
        feedback: str,
        outcome: str,
    ) -> HistoricalCase:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        case.clinician_decision = clinician_decision
        case.clinician_feedback = feedback
        case.outcome = outcome
        return case


class SimilarityScorer:
    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        raise NotImplementedError


class CosineSimilarityScorer(SimilarityScorer):
    """Deterministic numerical similarity for structured patient features."""

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Feature vectors must have the same length")
        left = [float(v) for v in left]
        right = [float(v) for v in right]
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class ExperienceMemory:
    """Experience memory that stores prior cases and retrieves similar ones."""

    def __init__(
        self,
        *,
        store: ExperienceStore | None = None,
        similarity_scorer: SimilarityScorer | None = None,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        self.store = store or InMemoryExperienceStore(
            similarity_scorer=similarity_scorer,
        )
        self.similarity_scorer = similarity_scorer or CosineSimilarityScorer()
        self.feature_names = tuple(feature_names)

    def store_case(self, case: HistoricalCase) -> HistoricalCase:
        return self.store.save_case(case)

    def get_case(self, case_id: str) -> HistoricalCase | None:
        return self.store.get_case(case_id)

    def retrieve_similar(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trend_result: TrendResult | Mapping[str, Any] | None,
        anomaly_result: Any,
        risk_result: Any,
        *,
        top_k: int = 5,
    ) -> ExperienceResult:
        feature_vector = build_case_feature_vector(
            patient_state,
            trend_result,
            anomaly_result,
            risk_result,
        )
        matches = self.store.search_similar(feature_vector, top_k=top_k)
        retrieved_cases = [case for case, _ in matches]
        similarity_scores = [score for _, score in matches]
        confidence = _calculate_experience_confidence(
            retrieved_cases, similarity_scores
        )
        explanation = (
            f"Retrieved {len(retrieved_cases)} similar historical cases with "
            f"mean similarity {sum(similarity_scores) / max(1, len(similarity_scores)):.3f}."
        )
        return ExperienceResult(
            retrieved_cases=retrieved_cases,
            experience_confidence=confidence,
            similarity_scores=similarity_scores,
            feature_vector=feature_vector,
            explanation=explanation,
        )

    def record_feedback(
        self,
        case_id: str,
        clinician_decision: str,
        feedback: str,
        outcome: str,
    ) -> HistoricalCase:
        return self.store.update_feedback(
            case_id,
            clinician_decision,
            feedback,
            outcome,
        )


def build_case_feature_vector(
    patient_state: PatientState | Mapping[str, Any],
    trend_result: TrendResult | Mapping[str, Any] | None,
    anomaly_result: Any,
    risk_result: Any,
) -> list[float]:
    vitals = (
        getattr(patient_state, "current_vitals", {})
        if patient_state is not None
        else {}
    )
    if not isinstance(vitals, Mapping):
        vitals = {}

    values = [
        _extract_feature_value(vitals, "HR", default=80.0),
        _extract_feature_value(vitals, "O2Sat", default=97.0),
        _extract_feature_value(vitals, "Resp", default=18.0),
        _extract_feature_value(vitals, "SBP", default=120.0),
        _extract_feature_value(vitals, "DBP", default=70.0),
        _extract_trend_score(trend_result),
        _extract_anomaly_score(anomaly_result),
        _extract_risk_score(risk_result),
    ]
    vital_names = ("HR", "O2Sat", "Resp", "SBP", "DBP")
    return [_normalize_feature(name, value) for name, value in zip(vital_names, values[:5])] + [
        float(values[5]),
        float(values[6]),
        float(values[7]),
    ]


def case_feature_vector(case: HistoricalCase) -> list[float]:
    pattern = case.physiological_pattern
    values = [
        float(pattern.get("HR", 80.0)),
        float(pattern.get("O2Sat", 97.0)),
        float(pattern.get("Resp", 18.0)),
        float(pattern.get("SBP", 120.0)),
        float(pattern.get("DBP", 70.0)),
        float(
            case.trend_information.get("score", 0.0)
            if isinstance(case.trend_information, Mapping)
            else 0.0
        ),
        float(case.anomaly_confidence),
        float(case.risk_score),
    ]
    vital_names = ("HR", "O2Sat", "Resp", "SBP", "DBP")
    return [_normalize_feature(name, value) for name, value in zip(vital_names, values[:5])] + [
        float(values[5]),
        float(values[6]),
        float(values[7]),
    ]


def _calculate_experience_confidence(
    retrieved_cases: Sequence[HistoricalCase],
    similarity_scores: Sequence[float],
) -> float:
    if not retrieved_cases:
        return 0.0

    mean_similarity = sum(similarity_scores) / max(1, len(similarity_scores))
    decision_signal = 0.0
    if retrieved_cases:
        numerator = 0.0
        for case in retrieved_cases:
            outcome = str(case.outcome or "").lower()
            if outcome in {"improved", "resolved", "stable", "observed"}:
                numerator += 1.0
        decision_signal = numerator / len(retrieved_cases)

    coverage = min(1.0, len(retrieved_cases) / 5.0)
    confidence = (0.6 * mean_similarity) + (0.25 * decision_signal) + (0.15 * coverage)
    return max(0.0, min(1.0, confidence))


def _extract_feature_value(
    vitals: Mapping[str, Any],
    vital_name: str,
    *,
    default: float,
) -> float:
    aliases = {
        "HR": ("HR", "heart_rate", "hr"),
        "O2Sat": ("O2Sat", "SpO2", "spo2", "oxygen_saturation"),
        "Resp": ("Resp", "RR", "respiratory_rate"),
        "SBP": ("SBP", "systolic_bp", "systolic"),
        "DBP": ("DBP", "diastolic_bp", "diastolic"),
    }
    for alias in aliases.get(vital_name, (vital_name,)):
        value = vitals.get(alias)
        for _ in range(3):
            if isinstance(value, Mapping):
                value = value.get("value")
            elif hasattr(value, "value"):
                value = value.value
            else:
                break
        if value is not None:
            return float(value)
    return float(default)


def _extract_trend_score(trend_result: TrendResult | Mapping[str, Any] | None) -> float:
    if trend_result is None:
        return 0.0
    if isinstance(trend_result, Mapping):
        value = trend_result.get("score", trend_result.get("trend_score", 0.0))
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))
    return max(0.0, min(1.0, float(getattr(trend_result, "score", 0.0))))


def _extract_anomaly_score(anomaly_result: Any) -> float:
    if anomaly_result is None:
        return 0.0
    if isinstance(anomaly_result, Mapping):
        value = anomaly_result.get("probability", anomaly_result.get("score", 0.0))
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))
    value = getattr(
        anomaly_result, "probability", getattr(anomaly_result, "score", 0.0)
    )
    return max(0.0, min(1.0, float(value)))


def _extract_risk_score(risk_result: Any) -> float:
    if risk_result is None:
        return 0.0
    if isinstance(risk_result, Mapping):
        value = risk_result.get("risk_score", 0.0)
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))
    value = getattr(risk_result, "risk_score", 0.0)
    return max(0.0, min(1.0, float(value)))


def _normalize_feature(name: str, value: float) -> float:
    ranges = {
        "HR": (40.0, 140.0),
        "O2Sat": (80.0, 100.0),
        "Resp": (8.0, 40.0),
        "SBP": (70.0, 180.0),
        "DBP": (40.0, 110.0),
    }
    low, high = ranges.get(name, (0.0, 1.0))
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))
