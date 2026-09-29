"""Shared trend result contract for P1 and downstream components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrendResult:
    """Deterministic trend-analysis result passed to P3."""

    patient_id: str
    timestamp: Any = None
    score: float = 0.0
    direction: str = "stable"
    contributing_signals: tuple[str, ...] = ()
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "score": self.score,
            "direction": self.direction,
            "contributing_signals": list(self.contributing_signals),
            "details": dict(self.details),
        }
