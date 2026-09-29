"""Stable, explicit JSON event envelopes used by file and Kafka transports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EventEnvelope:
    schema_version: str
    event_id: str
    event_time: str
    patient_id: str
    run_id: str
    source: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
