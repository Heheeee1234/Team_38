"""Connect the P1-P3 agents and expose an explicit P4 handoff."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from statistics import mean
from typing import Any, Mapping, Sequence
from pathlib import Path

from clinical_data_gen.anomaly import AnomalyAgent, AnomalyFeedback
from clinical_reasoning.adapters import ExperienceMemoryAdapter, evidence_from_p3
from clinical_reasoning.pipeline import P4Service
from clinical_data_gen.state import PatientState, PatientStateManager
from clinical_data_gen.trend import TrendResult
from experience import ExperienceMemory, HistoricalCase
from fusion import EvidenceFusion
from risk import RiskAssessor
from trend_agent.trend_agent import TrendAgent


_FORBIDDEN_LIVE_FIELDS = frozenset(
    {"scenario", "risk_label", "latent", "trend_phase", "episode_id"}
)
_VITAL_FIELDS = {
    "heart_rate": "HR",
    "spo2": "O2Sat",
    "temperature": "Temp",
    "systolic_bp": "SBP",
    "map": "MAP",
    "diastolic_bp": "DBP",
    "respiratory_rate": "Resp",
}
_TREND_GROUPS = {
    "heart_rate": "heart_rate",
    "spo2": "spo2",
    "respiratory_rate": "respiratory_rate",
    "systolic_bp": "blood_pressure",
    "diastolic_bp": "blood_pressure",
}


class DeteriorationWorkflow:
    """Stateful observed-data workflow; ground-truth labels are never accepted."""

    def __init__(self, *, baseline_window: int = 30) -> None:
        if baseline_window <= 0:
            raise ValueError("baseline_window must be positive")
        self.state_manager = PatientStateManager()
        self.trend_agent = TrendAgent()
        self.anomaly_agent = AnomalyAgent()
        self.risk_assessor = RiskAssessor()
        self.experience_memory = ExperienceMemory()
        self.evidence_fusion = EvidenceFusion()
        repository_root = Path(__file__).resolve().parents[2]
        self.clinical_reasoning = P4Service(
            repository_root / "knowledge" / "protocol_corpus.jsonl",
            live_llm=os.environ.get("P4_LIVE_LLM", "").strip().lower() == "true",
            experience_client=ExperienceMemoryAdapter(self.experience_memory),
            retrieval_backend=os.environ.get("P4_RETRIEVAL_BACKEND", "tfidf").strip().lower(),
            index_path=repository_root / "data" / "derived" / "p4_vector_index.sqlite3",
            embedding_model=os.environ.get(
                "P4_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            model=os.environ.get("P4_MODEL"),
        )
        self.baseline_window = baseline_window
        self.contexts: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self._states_by_event: dict[str, PatientState] = {}
        self._baseline_samples: dict[str, list[dict[str, float]]] = {}
        self._last_event_times: dict[str, datetime] = {}

    def process_context(self, event: Mapping[str, Any]) -> None:
        """Register a static patient context before replaying vital events."""
        patient_id = event.get("patient_id")
        payload = event.get("payload")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("Patient context must contain a non-empty patient_id")
        if not isinstance(payload, Mapping):
            raise ValueError("Patient context payload must be an object")
        self.contexts[patient_id] = dict(payload)
        self.trend_agent.observe_context({"patient_id": patient_id})

    def process_events(
        self,
        events: Sequence[Mapping[str, Any]],
        contexts: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, dict[str, Any]]:
        """Process a complete replay in event-time order and return outputs by event ID."""
        for context in contexts:
            self.process_context(context)
        ordered = sorted(events, key=lambda event: _parse_time(event.get("event_time")))
        for event in ordered:
            self.process_event(event)
        return self.results

    def process_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Run one observed event through state, trend, anomaly, risk and fusion."""
        _validate_live_event(event)
        event_id = str(event["event_id"])
        if event_id in self.results:
            return self.results[event_id]

        patient_id = str(event["patient_id"])
        timestamp = _parse_time(event["event_time"])
        previous_time = self._last_event_times.get(patient_id)
        if previous_time is not None and timestamp < previous_time:
            raise ValueError(f"Events for patient {patient_id} are not in event-time order")
        self._last_event_times[patient_id] = timestamp

        payload = event["payload"]
        observed_vitals = payload["vitals"]
        state_vitals = {
            target: _numeric_or_none(observed_vitals.get(source), source)
            for source, target in _VITAL_FIELDS.items()
        }
        context = self.contexts.get(patient_id, {})
        state = self.state_manager.update(
            patient_id=patient_id,
            timestamp=timestamp.isoformat(),
            vitals=state_vitals,
            static_context={
                "age": _numeric_or_none(context.get("age"), "age"),
                "gender": _gender_code(context.get("sex")),
            },
            metadata={
                "patient_context": deepcopy(context),
                "signal_quality": payload.get("quality", "unknown"),
                "source_event_id": event_id,
            },
        )
        baseline = self._update_baseline(patient_id, observed_vitals, state)

        trend_output = self.trend_agent.observe(dict(event))
        trend_result = _trend_result(trend_output)
        anomaly_result = self.anomaly_agent.predict(state)
        risk_result = self.risk_assessor.assess(state, trend_result, anomaly_result)
        experience_result = self.experience_memory.retrieve_similar(
            state,
            trend_result,
            anomaly_result,
            risk_result,
        )
        fused = self.evidence_fusion.fuse(
            state,
            trend_result,
            anomaly_result,
            risk_result,
            experience_result,
        )
        p4_evidence = evidence_from_p3(
            evidence_id=event_id,
            patient_id=patient_id,
            window_end=timestamp.isoformat(),
            current_vitals=observed_vitals,
            trend_result=trend_output,
            anomaly_result={
                "probability": anomaly_result.probability,
                "is_anomalous": anomaly_result.is_anomalous,
                "contributing_features": list(anomaly_result.contributing_features),
                "model_version": anomaly_result.model_version,
            },
            risk_result=risk_result,
            patient_context_payload=context,
            experience_result=experience_result,
            fusion_confidence=fused.final_confidence,
        )
        reasoning = self.clinical_reasoning.reason(p4_evidence).to_dict()

        case = HistoricalCase(
            case_id=event_id,
            patient_id=patient_id,
            timestamp=timestamp.isoformat(),
            physiological_pattern=_physiological_pattern(state),
            trend_information=trend_result.to_dict(),
            anomaly_confidence=anomaly_result.probability,
            risk_score=risk_result.risk_score,
            metadata={"model_version": anomaly_result.model_version},
        )
        self.experience_memory.store_case(case)

        output = {
            "event_id": event_id,
            "patient_id": patient_id,
            "event_time": timestamp.isoformat(),
            "baseline": dict(baseline),
            "trend": trend_output,
            "anomaly": {
                "probability": anomaly_result.probability,
                "is_anomalous": anomaly_result.is_anomalous,
                "contributing_features": list(anomaly_result.contributing_features),
                "model_version": anomaly_result.model_version,
            },
            "risk": risk_result.to_dict(),
            "experience": _experience_summary(experience_result),
            "fusion": {
                "risk_level": fused.risk_level,
                "risk_score": fused.risk_score,
                "online_confidence": fused.online_confidence,
                "experience_confidence": fused.experience_confidence,
                "final_confidence": fused.final_confidence,
                "evidence_summary": list(fused.evidence_summary),
            },
            "clinical_reasoning": reasoning,
        }
        self.results[event_id] = output
        self._states_by_event[event_id] = deepcopy(state)
        state.add_event(
            {"event_type": "agent_pipeline", "event_id": event_id,
             "model_version": anomaly_result.model_version}
        )
        return output

    def record_clinician_feedback(
        self,
        event_id: str,
        decision: str,
        reason: str = "",
        *,
        anomaly_label: bool | None = None,
    ) -> dict[str, Any]:
        """Update experience memory and only train when an explicit anomaly label exists."""
        state = self._states_by_event.get(event_id)
        if state is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        normalized_decision = decision.strip().lower()
        if normalized_decision not in {"accept", "dismiss", "defer", "investigate"}:
            raise ValueError(f"Unsupported clinician decision: {decision!r}")
        feedback = AnomalyFeedback(
            patient_id=state.patient_id,
            timestamp=state.timestamp,
            anomaly_label=anomaly_label,
            decision=normalized_decision,
            reason=reason,
        )
        loss = self.anomaly_agent.update(state, feedback)
        self.experience_memory.record_feedback(
            event_id,
            normalized_decision,
            reason,
            "unknown",
        )
        return {
            "event_id": event_id,
            "decision": normalized_decision,
            "anomaly_label": anomaly_label,
            "model_updated": loss is not None,
            "model_version": self.anomaly_agent.model_version,
        }

    def _update_baseline(
        self,
        patient_id: str,
        observed: Mapping[str, Any],
        state: PatientState,
    ) -> dict[str, float]:
        samples = self._baseline_samples.setdefault(patient_id, [])
        row = {
            target: float(observed[source])
            for source, target in _VITAL_FIELDS.items()
            if source in observed
            and observed[source] is not None
            and target in {"HR", "O2Sat", "Resp", "SBP", "DBP"}
        }
        if len(samples) < self.baseline_window:
            samples.append(row)
        if len(samples) >= self.baseline_window:
            names = set.intersection(*(set(sample) for sample in samples))
            baseline = {
                name: round(mean(sample[name] for sample in samples), 2)
                for name in names
            }
            for name, value in baseline.items():
                state.set_baseline(name, value)
            return baseline
        return {}


def _validate_live_event(event: Mapping[str, Any]) -> None:
    required = ("schema_version", "event_id", "event_time", "patient_id", "run_id", "source", "payload")
    missing = [field for field in required if field not in event]
    if missing:
        raise ValueError(f"Vital event is missing envelope fields: {', '.join(missing)}")
    leaked = _FORBIDDEN_LIVE_FIELDS.intersection(event)
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Vital event payload must be an object")
    leaked |= _FORBIDDEN_LIVE_FIELDS.intersection(payload)
    if leaked:
        raise ValueError(f"Evaluation-only fields in live event: {', '.join(sorted(leaked))}")
    vitals = payload.get("vitals")
    if not isinstance(vitals, Mapping) or not any(value is not None for value in vitals.values()):
        raise ValueError("Vital event payload must contain observed vitals")
    for key, value in vitals.items():
        _numeric_or_none(value, str(key))
    for field in ("event_id", "patient_id", "run_id", "source", "schema_version"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError(f"Vital event {field} must be a non-empty string")
    _parse_time(event["event_time"])


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("event_time must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid event_time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _numeric_or_none(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or null") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _gender_code(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"female", "f", "0"}:
        return 0
    if normalized in {"male", "m", "1"}:
        return 1
    return None


def _trend_result(output: Mapping[str, Any]) -> TrendResult:
    changes: dict[str, float] = {}
    groups: set[str] = set()
    for item in output["trends"]:
        name = item["vital"]
        changes[name] = float(item["change"])
        if item["direction"] != "stable":
            groups.add(_TREND_GROUPS[name])
    score = min(1.0, len(groups) / 3.0)
    return TrendResult(
        patient_id=str(output["patient_id"]),
        timestamp=output["event_time"],
        score=score,
        direction="worsening" if output["candidate_deterioration_trend"] else "stable",
        contributing_signals=tuple(output["concurrent_worsening_signals"]),
        details=changes,
    )


def _physiological_pattern(state: PatientState) -> dict[str, float]:
    pattern = {}
    for name, observation in state.current_vitals.items():
        value = observation
        for _ in range(3):
            if isinstance(value, Mapping):
                value = value.get("value")
            elif hasattr(value, "value"):
                value = value.value
            else:
                break
        if value is not None and not observation.is_missing:
            pattern[name] = float(value)
    return pattern


def _experience_summary(result: Any) -> dict[str, Any]:
    cases = []
    for case, similarity in zip(result.retrieved_cases, result.similarity_scores):
        cases.append({
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "decision": case.clinician_decision,
            "outcome": case.outcome,
            "similarity": similarity,
        })
    return {
        "experience_confidence": result.experience_confidence,
        "retrieved_cases": cases,
        "explanation": result.explanation,
    }
