"""Observed-data-only mock escalation policy for the clinician demo."""
from __future__ import annotations

from datetime import datetime
from typing import Any

CHANNELS = ("heart_rate", "spo2", "respiratory_rate", "systolic_bp", "diastolic_bp")
COOLDOWN_MINUTES = 30


def _time(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(event["event_time"])


def score_event(
    vitals: dict[str, float],
    baseline: dict[str, float],
    agent_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return simple patient-baseline and absolute-threshold evidence."""
    flags: list[str] = []
    critical: list[str] = []
    tests = (
        ("heart_rate", vitals["heart_rate"] >= max(120, baseline["heart_rate"] + 18), "Heart rate elevated from personal baseline"),
        ("spo2", vitals["spo2"] <= min(92, baseline["spo2"] - 3), "Oxygen saturation falling from personal baseline"),
        ("respiratory_rate", vitals["respiratory_rate"] >= max(24, baseline["respiratory_rate"] + 5), "Respiratory rate elevated from personal baseline"),
        ("respiratory_rate", vitals["respiratory_rate"] <= min(10, baseline["respiratory_rate"] - 5), "Respiratory rate is unusually low"),
        ("systolic_bp", vitals["systolic_bp"] <= min(95, baseline["systolic_bp"] - 20), "Systolic pressure falling from personal baseline"),
        ("fio2", vitals.get("fio2", 0) >= max(.30, .21 + .09), "Supplemental oxygen requirement increased"),
    )
    for channel, condition, label in tests:
        if condition and label not in flags:
            flags.append(label)
    flagged_channels = []
    if vitals["heart_rate"] >= max(120, baseline["heart_rate"] + 18): flagged_channels.append("heart_rate")
    if vitals["spo2"] <= min(92, baseline["spo2"] - 3): flagged_channels.append("spo2")
    if vitals["respiratory_rate"] >= max(24, baseline["respiratory_rate"] + 5) or vitals["respiratory_rate"] <= min(10, baseline["respiratory_rate"] - 5): flagged_channels.append("respiratory_rate")
    if vitals["systolic_bp"] <= min(95, baseline["systolic_bp"] - 20): flagged_channels.append("systolic_bp")
    if vitals.get("fio2", 0) >= .30: flagged_channels.append("fio2")
    absolute = (("heart_rate", vitals["heart_rate"] >= 140), ("spo2", vitals["spo2"] <= 85),
                ("respiratory_rate", vitals["respiratory_rate"] >= 30 or vitals["respiratory_rate"] <= 7),
                ("systolic_bp", vitals["systolic_bp"] <= 80))
    critical = [channel for channel, condition in absolute if condition]
    cross_system = len(set(flagged_channels)) >= 2 and any(name in flagged_channels for name in ("spo2", "systolic_bp", "fio2"))
    upstream = agent_outputs or {}
    trend = upstream.get("trend") or {}
    risk = upstream.get("risk") or {}
    anomaly = upstream.get("anomaly") or {}
    trend_candidate = bool(trend.get("candidate_deterioration_trend"))
    risk_score = float(risk.get("risk_score", 0.0))
    risk_level = str(risk.get("risk_level", "LOW")).upper()
    anomaly_probability = float(anomaly.get("probability", 0.0))
    upstream_eligible = risk_level in {"HIGH", "CRITICAL"} and trend_candidate
    if trend_candidate:
        flags.append("Trend analyzer detected sustained multi-vital change")
    if risk_level in {"HIGH", "CRITICAL"}:
        flags.append(f"Risk assessor: {risk_level} ({risk_score:.2f})")
    if anomaly_probability >= 0.75:
        flags.append(f"Anomaly confidence: {anomaly_probability:.2f}")
    return {"flags": flags, "flagged_channels": sorted(set(flagged_channels)), "critical_vitals": critical,
            "score": len(flags), "baseline_score": len(tests) and sum(
                condition for _, condition, _ in tests
            ), "baseline_flags": [
                label for _, condition, label in tests if condition
            ], "upstream_eligible": upstream_eligible,
            "upstream_evidence": {
                "trend_candidate": trend_candidate,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "anomaly_probability": anomaly_probability,
                "anomaly_alone_is_alert": False,
            },
            "alert_eligible": len([label for _, condition, label in tests if condition]) >= 3
            or cross_system or upstream_eligible}


def evaluate_patient(
    events: list[dict[str, Any]],
    *,
    cooldown_minutes: int = COOLDOWN_MINUTES,
    agent_outputs_by_event: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score an ordered patient stream and apply persistence, QC, priority, and cooldown."""
    events = sorted(events, key=_time)
    if not events:
        return []
    warmup = events[:min(30, len(events))]
    base = {key: sum(float(e["payload"]["vitals"][key]) for e in warmup) / len(warmup) for key in CHANNELS}
    decisions: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []
    last_alert_at: datetime | None = None
    last_priority = "Low"
    alert_episode_active = False
    quiet_readings = 0
    for index, event in enumerate(events):
        # Baseline-building observations must never count toward the later persistence rule.
        if index == 30:
            recent.clear()
        observed = event.get("payload", {}).get("vitals", {})
        if not all(key in observed for key in CHANNELS):
            continue
        upstream = (agent_outputs_by_event or {}).get(str(event.get("event_id")), {})
        evidence = score_event(observed, base, upstream)
        quality = event.get("payload", {}).get("quality", "unknown")
        persistence_evidence = evidence if quality != "suspect" else {"score": 0, "critical_vitals": [], "alert_eligible": False}
        recent = (recent + [persistence_evidence])[-3:]
        flags = evidence["flags"]
        persistent = sum(item.get("alert_eligible", False) for item in recent) >= 2
        critical_persistent = sum(bool(item["critical_vitals"]) for item in recent) >= 2 and any(
            len(set(item.get("critical_vitals", []))) >= 2 for item in recent
        )
        confirmed = persistent
        priority = "High" if (
            len(evidence["baseline_flags"]) >= 3
            or critical_persistent
            or evidence["upstream_eligible"]
        ) else "Medium"
        if quality == "suspect":
            policy, priority, reason = "SUPPRESS", "Low", "Signal quality is suspect; repeat measurement before escalation."
            quiet_readings = 0
        elif index < 30:
            policy, priority, reason = "DEFER", "Low", "Collect the initial patient baseline before evaluating trends."
            quiet_readings = 0
        elif confirmed:
            elapsed = ( _time(event) - last_alert_at).total_seconds() / 60 if last_alert_at else None
            upgraded = alert_episode_active and priority == "High" and last_priority != "High"
            if upgraded:
                policy = "ALERT"
                reason = "Priority upgraded after persistent evidence strengthened."
                last_alert_at, last_priority = _time(event), priority
            elif alert_episode_active:
                policy, priority, reason = "SUPPRESS", "Low", "Duplicate alert suppressed while the same episode remains active (cooldown)."
            elif elapsed is None or elapsed >= cooldown_minutes:
                policy = "ALERT"
                reason = "Persistent baseline or corroborated trend/risk evidence confirmed."
                last_alert_at, last_priority = _time(event), priority
                alert_episode_active = True
            else:
                policy, priority, reason = "SUPPRESS", "Low", f"Duplicate alert held during {cooldown_minutes}-minute cooldown."
            quiet_readings = 0
        elif flags:
            policy, priority, reason = "DEFER", "Low", "An isolated change is present; continue monitoring for persistence."
            quiet_readings = 0
        else:
            policy, priority, reason = "SUPPRESS", "Low", "No sustained multi-parameter change detected."
            quiet_readings += 1
            if quiet_readings >= cooldown_minutes:
                alert_episode_active = False
                last_priority = "Low"
        decisions.append({
            "event_id": event.get("event_id"), "event_time": event["event_time"],
            "patient_id": event["patient_id"], "observation": observed,
            "quality": quality, "baseline": {k: round(v, 2) for k, v in base.items()},
            "evidence": evidence, "persistence_confirmed": confirmed,
            "policy": policy, "priority": priority, "policy_reason": reason,
            "agent_outputs": upstream,
        })
    return decisions


def evaluate_stream(
    events: list[dict[str, Any]],
    agent_outputs_by_event: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_patient.setdefault(event["patient_id"], []).append(event)
    return {
        patient_id: evaluate_patient(
            rows,
            agent_outputs_by_event=agent_outputs_by_event,
        )
        for patient_id, rows in by_patient.items()
    }
