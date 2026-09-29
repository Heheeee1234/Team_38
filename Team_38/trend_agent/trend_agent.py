"""Stateful, explainable multi-parameter trend analysis for observed vital events.

This demonstration component identifies sustained directional changes. It is not a
clinical scoring system and does not diagnose or recommend treatment.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from statistics import mean
from typing import Any


VITAL_DIRECTIONS = {
    "heart_rate": ("rising", 10.0),
    "spo2": ("falling", 3.0),
    "respiratory_rate": ("rising", 4.0),
    "systolic_bp": ("falling", 10.0),
    "diastolic_bp": ("falling", 8.0),
}
SIGNAL_GROUP = {
    "heart_rate": "heart_rate",
    "spo2": "spo2",
    "respiratory_rate": "respiratory_rate",
    "systolic_bp": "blood_pressure",
    "diastolic_bp": "blood_pressure",
}


class TrendAgent:
    """Keep a bounded per-patient window and assess concurrent vital trends."""

    def __init__(self, window_size: int = 6, min_points: int = 4,
                 required_signals: int = 2) -> None:
        if window_size < min_points or min_points < 2 or required_signals < 1:
            raise ValueError("Require window_size >= min_points >= 2 and required_signals >= 1")
        self.window_size = window_size
        self.min_points = min_points
        self.required_signals = required_signals
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._seen: set[str] = set()

    def observe_context(self, event: dict[str, Any]) -> None:
        """Remember patient IDs with supplied static context (context is not streamed)."""
        if "patient_id" not in event:
            raise ValueError("Context event is missing patient_id")
        self._seen.add(str(event["patient_id"]))

    def observe(self, event: dict[str, Any]) -> dict[str, Any]:
        required = ("schema_version", "event_id", "event_time", "patient_id", "run_id", "source", "payload")
        missing = [field for field in required if field not in event]
        if missing:
            raise ValueError(f"Vital event missing required envelope fields: {', '.join(missing)}")
        patient_id = str(event["patient_id"])
        payload = event["payload"]
        vitals = payload.get("vitals") if isinstance(payload, dict) else None
        if not isinstance(vitals, dict):
            raise ValueError("Vital event payload must contain a vitals object")
        timestamp = datetime.fromisoformat(str(event["event_time"]).replace("Z", "+00:00"))
        row: dict[str, Any] = {"event_time": timestamp.isoformat()}
        for name in VITAL_DIRECTIONS:
            value = vitals.get(name)
            if value is not None:
                try:
                    row[name] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Vital {name} must be numeric") from exc
        if not any(name in row for name in VITAL_DIRECTIONS):
            raise ValueError("Vital event contains no supported vital values")
        history = self._history[patient_id]
        history.append(row)
        self._seen.add(patient_id)
        trends = self._analyze(list(history))
        worsening = [item for item in trends if item["direction"] != "stable"]
        worsening_groups = {SIGNAL_GROUP[item["vital"]] for item in worsening}
        candidate = len(worsening_groups) >= self.required_signals
        return {
            "schema_version": "1.0",
            "analysis_id": f"trend-{event['event_id']}",
            "event_time": event["event_time"],
            "patient_id": patient_id,
            "source_event_id": event["event_id"],
            "window_points": len(history),
            "ready": len(history) >= self.min_points,
            "candidate_deterioration_trend": candidate and len(history) >= self.min_points,
            "concurrent_worsening_signals": [item["vital"] for item in worsening],
            "trends": trends,
            "explanation": self._explain(worsening, len(history)),
            "clinical_context_available": patient_id in self._seen,
            "disclaimer": "Trend signal for review; not a diagnosis or clinical recommendation.",
        }

    def _analyze(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(rows) < 2:
            return []
        output = []
        for vital, (adverse_direction, threshold) in VITAL_DIRECTIONS.items():
            samples = [(i, row[vital]) for i, row in enumerate(rows) if vital in row]
            if len(samples) < 2:
                continue
            values = [value for _, value in samples]
            delta = values[-1] - values[0]
            adverse_delta = delta if adverse_direction == "rising" else -delta
            consistency = sum(
                (b - a) >= 0 if adverse_direction == "rising" else (b - a) <= 0
                for a, b in zip(values, values[1:])
            ) / max(1, len(values) - 1)
            direction = adverse_direction if adverse_delta >= threshold and consistency >= 0.6 else "stable"
            output.append({
                "vital": vital,
                "direction": direction,
                "start": round(values[0], 2),
                "latest": round(values[-1], 2),
                "change": round(delta, 2),
                "adverse_change_threshold": threshold,
                "directional_consistency": round(consistency, 2),
                "mean": round(mean(values), 2),
                "samples": len(values),
            })
        return output

    @staticmethod
    def _explain(worsening: list[dict[str, Any]], points: int) -> str:
        if points < 2:
            return "Waiting for additional readings before assessing a trend."
        if not worsening:
            return "No sustained multi-reading adverse change crossed the configured trend thresholds."
        details = "; ".join(
            f"{item['vital']} {item['direction']} by {abs(item['change']):g}"
            for item in worsening
        )
        return f"Observed sustained directional changes in {len(worsening)} signal(s): {details}."
