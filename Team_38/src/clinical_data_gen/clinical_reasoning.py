"""Legacy pending-result contract retained for compatibility with older callers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClinicalReasoningResult:
    status: str
    message: str
    explanation: str | None
    recommendation: str | None
    retrieved_evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "retrieved_evidence": list(self.retrieved_evidence),
            "citations": list(self.citations),
        }


class ClinicalReasoningPlaceholder:
    """Return the legacy pending status; the live workflow uses P4Service."""

    def run(self, evidence: Any) -> ClinicalReasoningResult:
        del evidence
        return ClinicalReasoningResult(
            status="not_implemented",
            message="Person 4's clinical knowledge retrieval and reasoning are pending.",
            explanation=None,
            recommendation=None,
            retrieved_evidence=[],
            citations=[],
        )
