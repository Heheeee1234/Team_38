import json
from datetime import datetime, timedelta, timezone

from clinician_app import server
from clinician_app.evaluation import calculate_metrics


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dashboard_and_audit_show_integrated_p4(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    events = []
    for minute in range(6):
        events.append({
            "schema_version": "1.0.0",
            "event_id": f"dashboard-event-{minute}",
            "event_time": (start + timedelta(minutes=minute)).isoformat(),
            "patient_id": "SYN-0001",
            "run_id": "dashboard-test",
            "source": "test-monitor",
            "payload": {
                "vitals": {
                    "heart_rate": 130 + minute,
                    "spo2": 86 - minute,
                    "respiratory_rate": 30 + minute,
                    "systolic_bp": 90 - minute,
                    "diastolic_bp": 60,
                    "map": 65,
                    "fio2": 0.3,
                },
                "quality": "normal",
            },
        })
    contexts = [{
        "patient_id": "SYN-0001",
        "payload": {"age": 72, "sex": "female", "relevant_history": ["COPD"]},
    }]
    _write_jsonl(generated / "vitals_observed.jsonl", events)
    _write_jsonl(generated / "patient_context.jsonl", contexts)
    _write_jsonl(generated / "ground_truth_episodes.jsonl", [])
    _write_jsonl(generated / "evaluation_labels.jsonl", [])

    monkeypatch.setattr(server, "DATA", generated)
    monkeypatch.setattr(server, "AUDIT", tmp_path / "audit.jsonl")
    monkeypatch.setattr(server, "FEEDBACK", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(server, "_WORKFLOW", None)
    monkeypatch.setattr(server, "_WORKFLOW_CACHE_KEY", None)
    monkeypatch.setenv("P4_RETRIEVAL_BACKEND", "tfidf")
    monkeypatch.setenv("P4_LIVE_LLM", "false")

    result = server.dashboard()

    reasoning = result["cases"][0]["clinical_reasoning"]
    assert result["mode"] == "Integrated demo · P4 active"
    assert reasoning["model"] == "deterministic-p4"
    assert reasoning["recommended_action"]
    assert reasoning["citations"]
    audit = server.read_jsonl(server.AUDIT)
    assert audit[0]["citations"]
    assert audit[0]["reasoning"]
    assert audit[0]["recommendation"]


def test_citation_coverage_counts_cited_audited_alerts():
    decisions = {
        "p1": [{
            "policy": "ALERT",
            "event_id": "e1",
            "patient_id": "p1",
            "event_time": "2026-09-18T00:00:00+00:00",
        }],
    }
    audit = [{"alert": {"decision": "ALERT"}, "citations": [{"document_id": "news2"}]}]

    metrics, _ = calculate_metrics(decisions, [], {}, audit)

    assert metrics["rag_citation_coverage"] == 1.0
