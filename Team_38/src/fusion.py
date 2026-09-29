from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfidenceFusionConfig:
    alpha: float = 0.7


class ConfidenceFusion:
    """Combine online and experiential confidence using the specified weighting."""

    def __init__(self, *, alpha: float = 0.7) -> None:
        self.alpha = alpha

    def combine(
        self,
        online_confidence: float,
        experience_confidence: float,
        *,
        alpha: float | None = None,
    ) -> float:
        alpha_value = self.alpha if alpha is None else float(alpha)
        if not 0.0 <= alpha_value <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        online_value = _validate_confidence(online_confidence, "online_confidence")
        experience_value = _validate_confidence(
            experience_confidence,
            "experience_confidence",
        )
        final = alpha_value * online_value + (1.0 - alpha_value) * experience_value
        return max(0.0, min(1.0, final))


@dataclass(frozen=True)
class FusedEvidence:
    patient_id: str
    timestamp: Any
    risk_level: str
    risk_score: float
    online_confidence: float
    experience_confidence: float
    final_confidence: float
    evidence_summary: list[str] = field(default_factory=list)
    supporting_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "online_confidence": self.online_confidence,
            "experience_confidence": self.experience_confidence,
            "final_confidence": self.final_confidence,
            "evidence_summary": list(self.evidence_summary),
            "supporting_evidence": dict(self.supporting_evidence),
        }


class EvidenceFusion:
    """Aggregate the patient state, trend, anomaly, risk, and memory evidence."""

    def __init__(self, *, fusion: ConfidenceFusion | None = None) -> None:
        self.fusion = fusion or ConfidenceFusion()

    def fuse(
        self,
        patient_state: Any,
        trend_result: Any,
        anomaly_result: Any,
        risk_result: Any,
        experience_result: Any,
        final_confidence: float | None = None,
    ) -> FusedEvidence:
        patient_id = getattr(patient_state, "patient_id", "unknown")
        timestamp = getattr(patient_state, "timestamp", None)
        risk_level = getattr(risk_result, "risk_level", "LOW")
        risk_score = getattr(risk_result, "risk_score", 0.0)

        online_confidence = _extract_confidence(anomaly_result, "probability")
        experience_confidence = getattr(
            experience_result,
            "experience_confidence",
            0.0,
        )
        if final_confidence is None:
            final_confidence = self.fusion.combine(
                online_confidence,
                experience_confidence,
            )
        else:
            final_confidence = _validate_confidence(
                final_confidence,
                "final_confidence",
            )

        evidence_summary = [
            f"risk={risk_level}",
            f"online_confidence={online_confidence:.3f}",
            f"experience_confidence={experience_confidence:.3f}",
            f"final_confidence={final_confidence:.3f}",
        ]
        if trend_result is not None:
            trend_score = getattr(trend_result, "score", 0.0)
            evidence_summary.append(f"trend_score={float(trend_score):.3f}")

        supporting_evidence = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "risk_factors": getattr(risk_result, "contributing_factors", []),
            "trend_score": (
                getattr(trend_result, "score", 0.0) if trend_result is not None else 0.0
            ),
            "online_confidence": online_confidence,
            "experience_confidence": experience_confidence,
            "final_confidence": final_confidence,
            "retrieved_cases": getattr(experience_result, "retrieved_cases", []),
        }

        return FusedEvidence(
            patient_id=str(patient_id),
            timestamp=timestamp,
            risk_level=str(risk_level),
            risk_score=float(risk_score),
            online_confidence=float(online_confidence),
            experience_confidence=float(experience_confidence),
            final_confidence=float(final_confidence),
            evidence_summary=evidence_summary,
            supporting_evidence=supporting_evidence,
        )


def _validate_confidence(value: float, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return score


def _extract_confidence(value: Any, attr_name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Mapping):
        score = value.get(attr_name, 0.0)
    else:
        score = getattr(value, attr_name, 0.0)
    return _validate_confidence(score, attr_name)


def combine(
    online_confidence: float,
    experience_confidence: float,
    alpha: float = 0.7,
) -> float:
    return ConfidenceFusion(alpha=alpha).combine(
        online_confidence,
        experience_confidence,
    )
