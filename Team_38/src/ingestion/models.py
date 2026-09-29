"""Validated event contracts for the Kafka ingestion path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping

SUPPORTED_VITALS = frozenset({
    "heart_rate",
    "spo2",
    "temperature",
    "systolic_bp",
    "map",
    "diastolic_bp",
    "respiratory_rate",
    "fio2",
})
TREND_VITALS = frozenset({
    "heart_rate",
    "spo2",
    "respiratory_rate",
    "systolic_bp",
    "diastolic_bp",
})
EVENT_ENVELOPE_FIELDS = (
    "schema_version",
    "event_id",
    "event_time",
    "patient_id",
    "run_id",
    "source",
    "payload",
)


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("event_time must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid event_time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_context_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("patient context event must be a JSON object")
    missing = [field for field in EVENT_ENVELOPE_FIELDS if field not in value]
    if missing:
        raise ValueError(f"patient context is missing fields: {', '.join(missing)}")
    for field in ("schema_version", "event_id", "patient_id", "run_id", "source"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"patient context {field} must be a non-empty string")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("patient context payload must be a JSON object")
    event = dict(value)
    event["event_time"] = _parse_timestamp(value["event_time"]).isoformat().replace("+00:00", "Z")
    return event


@dataclass(frozen=True)
class VitalEvent:
    """Generator-compatible observed-vitals event."""

    event_id: str
    patient_id: str
    event_time: datetime
    payload: Mapping[str, Any]
    source_event: Mapping[str, Any]

    @classmethod
    def model_validate(cls, value: Any) -> "VitalEvent":
        if not isinstance(value, Mapping):
            raise ValueError("event must be a JSON object")
        for key in ("event_id", "patient_id"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f"{key} must be a non-empty string")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a JSON object")
        vitals = payload.get("vitals")
        if not isinstance(vitals, Mapping):
            raise ValueError("payload.vitals must be a JSON object")
        unknown = set(vitals).difference(SUPPORTED_VITALS)
        if unknown:
            raise ValueError(f"Unsupported vital fields: {', '.join(sorted(unknown))}")
        for name, reading in vitals.items():
            if reading is not None and (
                isinstance(reading, bool)
                or not isinstance(reading, (int, float))
                or not math.isfinite(reading)
            ):
                raise ValueError(f"payload.vitals.{name} must be a finite number or null")
        if not any(vitals.get(name) is not None for name in TREND_VITALS):
            raise ValueError("payload.vitals must contain an observed trend vital")
        return cls(
            event_id=value["event_id"],
            patient_id=value["patient_id"],
            event_time=_parse_timestamp(value.get("event_time")),
            payload=dict(payload),
            source_event=dict(value),
        )

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        event_time = (
            self.event_time.isoformat().replace("+00:00", "Z")
            if mode == "json"
            else self.event_time
        )
        return {
            **dict(self.source_event),
            "event_time": event_time,
            "payload": dict(self.payload),
        }

    @property
    def _vitals(self) -> Mapping[str, Any]:
        return self.payload["vitals"]

    @property
    def heart_rate(self) -> float | None:
        return self._vitals.get("heart_rate")

    @property
    def spo2(self) -> float | None:
        return self._vitals.get("spo2")

    @property
    def respiratory_rate(self) -> float | None:
        return self._vitals.get("respiratory_rate")

    @property
    def systolic_bp(self) -> float | None:
        return self._vitals.get("systolic_bp")

    @property
    def diastolic_bp(self) -> float | None:
        return self._vitals.get("diastolic_bp")

    @property
    def temperature(self) -> float | None:
        return self._vitals.get("temperature")

    @property
    def mean_arterial_pressure(self) -> float | None:
        return self._vitals.get("map")

    @property
    def fio2(self) -> float | None:
        return self._vitals.get("fio2")
