from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clinical_data_gen.pipeline import DeteriorationWorkflow
from clinician_app.policy import score_event


def vital_event(event_id: str, minute: int, *, patient_id: str = "patient-1") -> dict:
    return {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "event_time": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)).isoformat(),
        "patient_id": patient_id,
        "run_id": "test-run",
        "source": "test-monitor",
        "payload": {
            "vitals": {
                "heart_rate": 82 + minute,
                "spo2": 97 - min(minute, 5),
                "respiratory_rate": 17 + minute,
                "systolic_bp": 120 - minute,
                "diastolic_bp": 75 - minute / 2,
                "map": 85,
                "fio2": 0.21,
            },
            "quality": "normal",
        },
    }


def test_pipeline_connects_p1_through_grounded_p4_reasoning():
    workflow = DeteriorationWorkflow()
    workflow.process_context({
        "patient_id": "patient-1",
        "payload": {"age": 67, "sex": "female", "relevant_history": ["COPD"]},
    })

    outputs = [workflow.process_event(vital_event(f"event-{i}", i)) for i in range(6)]

    assert outputs[-1]["trend"]["patient_id"] == "patient-1"
    assert "probability" in outputs[-1]["anomaly"]
    assert outputs[-1]["risk"]["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert "final_confidence" in outputs[-1]["fusion"]
    reasoning = outputs[-1]["clinical_reasoning"]
    assert reasoning["model"] == "deterministic-p4"
    assert reasoning["explanation"]
    assert reasoning["recommended_action"]
    assert reasoning["citations"]
    assert all(
        citation["document_id"] and citation["version"] and citation["locator"]
        for citation in reasoning["citations"]
    )
    assert any(
        factor.startswith("anomaly probability ")
        for factor in reasoning["contributing_factors"]
    )


def test_clinician_action_does_not_become_anomaly_label():
    workflow = DeteriorationWorkflow()
    event = vital_event("event-feedback", 0)
    workflow.process_event(event)

    result = workflow.record_clinician_feedback("event-feedback", "Dismiss", "Not actionable.")

    assert result["anomaly_label"] is None
    assert result["model_updated"] is False
    assert result["model_version"] == 0


def test_live_pipeline_rejects_evaluation_labels():
    event = vital_event("event-leak", 0)
    event["payload"]["scenario"] = "sepsis_like"

    with pytest.raises(ValueError, match="Evaluation-only fields"):
        DeteriorationWorkflow().process_event(event)


def test_anomaly_alone_does_not_make_an_alert_eligible():
    vitals = {
        "heart_rate": 80,
        "spo2": 97,
        "respiratory_rate": 18,
        "systolic_bp": 120,
        "diastolic_bp": 70,
    }
    baseline = dict(vitals)

    result = score_event(vitals, baseline, {
        "anomaly": {"probability": 0.99, "is_anomalous": True},
        "risk": {"risk_level": "LOW", "risk_score": 0.1},
        "trend": {"candidate_deterioration_trend": False},
    })

    assert result["alert_eligible"] is False
    assert result["upstream_evidence"]["anomaly_alone_is_alert"] is False
