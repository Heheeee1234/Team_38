"""Episode-level evaluation for the Person 5 alert policy.

Ground-truth labels are consumed here only; the policy accepts observed events only.
"""
from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any


def calculate_metrics(
    decisions_by_patient: dict[str, list[dict[str, Any]]],
    episodes: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    audit_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positive = episodes
    alerts = [d for stream in decisions_by_patient.values() for d in stream if d["policy"] == "ALERT"]
    event_decisions = {d["event_id"]: d for stream in decisions_by_patient.values() for d in stream if d.get("event_id")}
    matched: set[str] = set()
    leads: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    for episode in positive:
        onset = datetime.fromisoformat(episode["onset_time"])
        window = datetime.fromisoformat(episode["escalation_window_start"])
        patient_alerts = sorted((a for a in alerts if a["patient_id"] == episode["patient_id"]), key=lambda x: x["event_time"])
        hits = [a for a in patient_alerts if onset <= datetime.fromisoformat(a["event_time"]) <= window]
        first = hits[0] if hits else None
        if first:
            matched.add(episode["patient_id"])
            leads.append((window - datetime.fromisoformat(first["event_time"])).total_seconds() / 60)
        prediction_rows.append({
            "episode_id": episode["episode_id"], "patient_id": episode["patient_id"],
            "scenario": episode["scenario"], "onset_time": episode["onset_time"],
            "escalation_window_start": episode["escalation_window_start"],
            "detected_in_window": bool(first), "first_alert_time": first["event_time"] if first else None,
            "lead_time_minutes": round(leads[-1], 2) if first else None,
        })
    true_positive_episodes = len(matched)
    episode_windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for episode in positive:
        episode_windows.setdefault(episode["patient_id"], []).append((
            datetime.fromisoformat(episode["onset_time"]),
            datetime.fromisoformat(episode["escalation_window_start"]),
        ))
    # Group alerts for one patient that occur within the policy cooldown into one alert episode.
    alert_clusters: list[list[dict[str, Any]]] = []
    for patient_id in sorted({a["patient_id"] for a in alerts}):
        rows = sorted((a for a in alerts if a["patient_id"] == patient_id), key=lambda x: x["event_time"])
        patient_clusters: list[list[dict[str, Any]]] = []
        for alert in rows:
            when = datetime.fromisoformat(alert["event_time"])
            if not patient_clusters or (when - datetime.fromisoformat(patient_clusters[-1][-1]["event_time"])).total_seconds() / 60 > 30:
                patient_clusters.append([alert])
            else:
                patient_clusters[-1].append(alert)
        alert_clusters.extend(patient_clusters)
    matched_alert_clusters = sum(any(
        start <= datetime.fromisoformat(alert["event_time"]) <= end
        for alert in cluster
        for start, end in episode_windows.get(cluster[0]["patient_id"], [])
    ) for cluster in alert_clusters)
    false_alert_events = sum(1 for a in alerts if not any(
        start <= datetime.fromisoformat(a["event_time"]) <= end
        for start, end in episode_windows.get(a["patient_id"], [])
    ))
    false_alert_clusters = len(alert_clusters) - matched_alert_clusters
    precision = matched_alert_clusters / len(alert_clusters) if alert_clusters else 0.0
    recall = true_positive_episodes / len(positive) if positive else 0.0
    total_hours = 0.0
    for stream in decisions_by_patient.values():
        if len(stream) > 1:
            total_hours += max(0, (datetime.fromisoformat(stream[-1]["event_time"]) - datetime.fromisoformat(stream[0]["event_time"])).total_seconds()) / 3600
    trigger_attempts = [d for stream in decisions_by_patient.values() for d in stream if d.get("persistence_confirmed") and d.get("quality") != "suspect"]
    suppressed = sum(d["policy"] == "SUPPRESS" and ("cooldown" in d["policy_reason"].lower() or "duplicate alert" in d["policy_reason"].lower()) for d in trigger_attempts)
    raised_attempts = sum(d["policy"] == "ALERT" for d in trigger_attempts)
    suppression_rate = suppressed / (suppressed + raised_attempts) if suppressed + raised_attempts else 0.0
    alert_traces = [row for row in (audit_records or [])
                    if (row.get("alert") or {}).get("decision") == "ALERT"]
    cited_alerts = sum(bool(row.get("citations")) for row in alert_traces)
    citation_coverage = cited_alerts / len(alert_traces) if alert_traces else 0.0
    artifact_event_ids = {event_id for event_id, label in labels.items() if label.get("artifact_injected")}
    artifact_alerts = sum(1 for event_id in artifact_event_ids if event_decisions.get(event_id, {}).get("policy") == "ALERT")
    artifact_fpr = artifact_alerts / len(artifact_event_ids) if artifact_event_ids else None
    trace = audit_records or []
    required_trace_fields = ("timestamp", "event_type", "patient_id", "observation", "agent_outputs", "retrieved_evidence", "citations", "reasoning", "alert")
    complete = sum(all(field in row for field in required_trace_fields) for row in trace)
    metrics = {
        "scope": "Generated synthetic cohort; observed-data-only alert policy; patient-level episode evaluation",
        "episodes": len(positive), "detected_episodes": true_positive_episodes,
        "alert_events": len(alerts), "alert_episodes": len(alert_clusters),
        "matched_alert_episodes": matched_alert_clusters, "false_alert_episodes": false_alert_clusters,
        "false_alert_events": false_alert_events,
        "episode_precision": round(precision, 4), "episode_recall": round(recall, 4),
        "median_lead_time_minutes": round(median(leads), 2) if leads else None,
        "alerts_per_patient_hour": round(len(alerts) / total_hours, 4) if total_hours else 0.0,
        "duplicate_suppression_rate": round(suppression_rate, 4),
        "artifact_false_positive_rate": round(artifact_fpr, 4) if artifact_fpr is not None else None,
        "artifact_alerts": artifact_alerts, "artifact_observations": len(artifact_event_ids),
        "rag_citation_coverage": round(citation_coverage, 4),
        "audit_record_completeness": round(complete / len(trace), 4) if trace else 0.0,
        "audit_records_evaluated": len(trace),
        "alert_policy": {"minimum_persistent_flags": 2, "persistence_window_readings": 3, "requires_cross_system_evidence": True, "cooldown_minutes": 30,
            "relative_thresholds": {"heart_rate_delta": 18, "spo2_drop": 3, "respiratory_rate_delta": 5, "respiratory_rate_low_drop": 5, "systolic_pressure_drop": 20},
            "uses_ground_truth": False},
        "definitions": {
            "episode_match": "At least one alert between labelled onset and escalation-window start, inclusive.",
            "precision": "Alert episodes (alerts for one patient within a 30-minute interval are clustered) that overlap a labelled onset-to-escalation interval divided by all alert episodes.",
            "recall": "Episodes with an alert in the matching window divided by all labelled positive episodes.",
            "duplicate_suppression": "Confirmed trigger candidates suppressed by cooldown divided by candidates either suppressed or raised.",
            "artifact_false_positive": "Artifact-labelled observations receiving an alert divided by all artifact-labelled observations.",
            "rag_citation_coverage": "Alert policy evaluations with at least one retrieved protocol citation divided by audited alert evaluations.",
        },
    }
    return metrics, prediction_rows
