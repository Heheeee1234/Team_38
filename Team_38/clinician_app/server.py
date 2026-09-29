"""Local Person 5 clinician interface, observed-only policy, and audit endpoints."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from clinician_app.evaluation import calculate_metrics
from clinician_app.policy import CHANNELS, evaluate_stream
from clinical_data_gen.pipeline import DeteriorationWorkflow

DATA = ROOT / "data" / "generated"
AUDIT = ROOT / "data" / "evaluation" / "clinician_audit.jsonl"
FEEDBACK = ROOT / "data" / "evaluation" / "clinician_feedback.jsonl"
PORT = int(os.environ.get("CLINICIAN_APP_PORT", "8765"))
POLICY_VERSION = "integrated-p4-policy-v8"
_WORKFLOW_CACHE_KEY: tuple[str, int, str] | None = None
_WORKFLOW: DeteriorationWorkflow | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def input_data() -> tuple[list[dict[str, Any]], dict[str, dict], list[dict], dict[str, dict]]:
    events = read_jsonl(DATA / "vitals_observed.jsonl")
    contexts = {x["patient_id"]: x.get("payload", {}) for x in read_jsonl(DATA / "patient_context.jsonl")}
    episodes = read_jsonl(DATA / "ground_truth_episodes.jsonl")
    labels = {x["event_id"]: x for x in read_jsonl(DATA / "evaluation_labels.jsonl")}
    return events, contexts, episodes, labels


def workflow_for(
    events: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
) -> DeteriorationWorkflow:
    """Build the observed-only agent workflow once for each generated cohort."""
    global _WORKFLOW_CACHE_KEY, _WORKFLOW
    if not events:
        raise ValueError("Cannot initialize the workflow without vital events")
    cache_key = (
        str(events[0].get("run_id", "")),
        len(events),
        str(events[-1].get("event_id", "")),
    )
    if _WORKFLOW is None or _WORKFLOW_CACHE_KEY != cache_key:
        workflow = DeteriorationWorkflow()
        context_events = [
            {"patient_id": patient_id, "payload": payload}
            for patient_id, payload in contexts.items()
        ]
        for context in context_events:
            workflow.process_context(context)
        decisions_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in read_jsonl(AUDIT):
            if record.get("event_type") == "clinician_decision" and record.get("event_id"):
                decisions_by_event[str(record["event_id"])].append(record)
        def event_time_key(event: dict[str, Any]) -> datetime:
            timestamp = datetime.fromisoformat(
                str(event["event_time"]).replace("Z", "+00:00")
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)

        ordered_events = sorted(events, key=event_time_key)
        for event in ordered_events:
            event_id = str(event["event_id"])
            workflow.process_event(event)
            for decision in decisions_by_event.get(event_id, []):
                workflow.record_clinician_feedback(
                    event_id,
                    str(decision.get("action", "Investigate")),
                    str(decision.get("reason", "")),
                )
        _WORKFLOW, _WORKFLOW_CACHE_KEY = workflow, cache_key
    return _WORKFLOW


def trend_summary(history: list[dict[str, Any]], key: str) -> float:
    prior, recent = history[-20:-10], history[-10:]
    if not prior or not recent:
        return 0.0
    mean = lambda rows: sum(float(x.get(key, 0)) for x in rows) / len(rows)
    return round(mean(recent) - mean(prior), 1)


def make_audit_traces(
    events: list[dict],
    decisions: dict[str, list[dict]],
    contexts: dict[str, dict],
    agent_outputs_by_event: dict[str, dict[str, Any]] | None = None,
) -> list[dict]:
    """Create one idempotent policy/observation trace per generated reading."""
    old_audit = read_jsonl(AUDIT)
    current_event_ids = {x.get("event_id") for x in events}
    stale = [x for x in old_audit if x.get("event_type") == "policy_evaluation"
             and x.get("event_id") in current_event_ids and x.get("policy_version") != POLICY_VERSION]
    if stale:
        # Replace obsolete outputs from an earlier version of this mock policy while preserving clinician actions.
        stale_ids = {x.get("event_id") for x in stale}
        preserved = [x for x in old_audit if not (x.get("event_type") == "policy_evaluation" and x.get("event_id") in stale_ids)]
        temporary = AUDIT.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preserved), encoding="utf-8")
        temporary.replace(AUDIT)
        old_audit = preserved
    existing = {x.get("event_id") for x in old_audit if x.get("event_type") == "policy_evaluation" and x.get("policy_version") == POLICY_VERSION}
    traces = []
    event_groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        event_groups[event["patient_id"]].append(event)
    for patient_id, rows in event_groups.items():
        rows.sort(key=lambda x: x["event_time"])
        decision_by_id = {d.get("event_id"): d for d in decisions.get(patient_id, [])}
        recent = []
        for event in rows:
            event_id = event.get("event_id")
            if event_id in existing:
                recent = (recent + [{"time": event["event_time"], **event.get("payload", {}).get("vitals", {})}])[-20:]
                continue
            decision = decision_by_id.get(event_id)
            if not decision:
                continue
            payload = event.get("payload", {})
            observed = payload.get("vitals", {})
            recent = (recent + [{"time": event["event_time"], **observed}])[-20:]
            trend = {name: trend_summary(recent, name) for name in CHANNELS}
            agent_outputs = (agent_outputs_by_event or {}).get(str(event_id), {})
            traces.append({
                "event_type": "policy_evaluation", "timestamp": event["event_time"],
                "event_id": event_id, "patient_id": patient_id, "policy_version": POLICY_VERSION,
                "run_id": event.get("run_id"),
                "observation": {"vitals": observed, "quality": payload.get("quality", "unknown"), "schema_version": event.get("schema_version")},
                "patient_context": contexts.get(patient_id, {}),
                "agent_outputs": {
                    "person1_trend": agent_outputs.get("trend"),
                    "person2_anomaly": agent_outputs.get("anomaly"),
                    "person3_risk": agent_outputs.get("risk"),
                    "person3_experience": agent_outputs.get("experience"),
                    "person3_fusion": agent_outputs.get("fusion"),
                    "person4_clinical_reasoning": agent_outputs.get("clinical_reasoning"),
                    "trend_summary_demo": trend,
                    "person5_alert_policy": decision["evidence"],
                },
                "retrieved_evidence": (agent_outputs.get("clinical_reasoning") or {}).get("citations", []),
                "citations": (agent_outputs.get("clinical_reasoning") or {}).get("citations", []),
                "reasoning": (agent_outputs.get("clinical_reasoning") or {}).get("explanation"),
                "recommendation": (agent_outputs.get("clinical_reasoning") or {}).get("recommended_action"),
                "alert": {"decision": decision["policy"], "priority": decision["priority"], "evidence": decision["evidence"].get("flags", [])},
                "source": "generated_mock_replay", "mock_data": True,
            })
    append_jsonl(AUDIT, traces)
    return traces


def dashboard() -> dict[str, Any]:
    events, contexts, episodes, labels = input_data()
    if not events:
        return {"cases": [], "metrics": {}, "audit": [], "mode": "Integrated demo · P4 active", "patient_count": 0,
                "error": "No generated mock data found. Generate a cohort first, then refresh."}
    workflow = workflow_for(events, contexts)
    agent_outputs = workflow.results
    decisions = evaluate_stream(events, agent_outputs)
    make_audit_traces(events, decisions, contexts, agent_outputs)
    event_groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        event_groups[event["patient_id"]].append(event)
    action_by_patient = {}
    clinician_records = [x for x in read_jsonl(AUDIT) if x.get("event_type") == "clinician_decision"]
    for row in clinician_records:
        action_by_patient[row.get("patient_id")] = row
    cases = []
    for patient_id, stream in event_groups.items():
        stream.sort(key=lambda x: x["event_time"])
        latest = stream[-1]
        latest_decision = next((x for x in reversed(decisions[patient_id]) if x.get("event_id") == latest.get("event_id")), decisions[patient_id][-1])
        latest_agents = agent_outputs[latest["event_id"]]
        history = [{"time": x["event_time"], **x["payload"]["vitals"]} for x in stream[-30:]]
        context = contexts.get(patient_id, {})
        baseline = latest_decision["baseline"]
        flags = latest_decision["evidence"].get("flags", [])
        cases.append({
            "patient_id": patient_id, "event_id": latest.get("event_id"), "event_time": latest["event_time"],
            "vitals": latest["payload"]["vitals"], "baseline": baseline, "history": history, "context": context,
            "priority": latest_decision["priority"], "policy": latest_decision["policy"],
            "policy_reason": latest_decision["policy_reason"], "factors": flags or ["No sustained multi-parameter change detected"],
            "explanation": latest_agents["trend"]["explanation"],
            "recommendation": latest_agents["clinical_reasoning"]["recommended_action"],
            "quality": latest.get("payload", {}).get("quality", "unknown"),
            "agents": latest_agents,
            "clinical_reasoning": latest_agents["clinical_reasoning"],
            "last_action": action_by_patient.get(patient_id),
        })
    cases.sort(key=lambda x: ({"High": 0, "Medium": 1, "Low": 2}.get(x["priority"], 3), x["patient_id"]))
    traces = [x for x in read_jsonl(AUDIT) if x.get("event_type") == "policy_evaluation"]
    metrics, _ = calculate_metrics(decisions, episodes, labels, traces)
    return {"cases": cases, "metrics": metrics, "audit": clinician_records[-12:],
            "mode": "Integrated demo · P4 active",
            "patient_count": len(event_groups)}


CLINICIAN_ACTIONS = ("Accept", "Dismiss", "Defer", "Investigate")


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/api/dashboard":
            self.send_json(dashboard())
            return
        if self.path == "/":
            raw = (ROOT / "clinician_app" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return
        if not isinstance(body, dict):
            self.send_json({"error": "Request body must be a JSON object"}, 400)
            return
        if self.path == "/api/action":
            action, patient_id = body.get("action"), body.get("patient_id")
            if action not in {"Accept", "Dismiss", "Defer", "Investigate"} or not patient_id:
                self.send_json({"error": "A patient and valid action are required"}, 400)
                return
            current = next((case for case in dashboard()["cases"] if case["patient_id"] == patient_id), None)
            if current is None:
                self.send_json({"error": "Patient is not in the current mock cohort"}, 404)
                return
            now = datetime.now().astimezone().isoformat()
            reason = str(body.get("reason", ""))[:1000]
            events, contexts, _, _ = input_data()
            workflow = workflow_for(events, contexts)
            try:
                learning_update = workflow.record_clinician_feedback(
                    current["event_id"],
                    action,
                    reason,
                )
            except (KeyError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            agent_outputs = current["agents"]
            decision = {"event_type": "clinician_decision", "timestamp": now, "patient_id": patient_id,
                        "event_id": current["event_id"], "action": action, "reason": reason,
                        "observation": current["vitals"], "agent_outputs": {
                            "person1_trend": agent_outputs["trend"],
                            "person2_anomaly": agent_outputs["anomaly"],
                            "person3_risk": agent_outputs["risk"],
                            "person3_experience": agent_outputs["experience"],
                            "person3_fusion": agent_outputs["fusion"],
                            "person4_clinical_reasoning": agent_outputs["clinical_reasoning"],
                            "person5_alert_policy": {"decision": current["policy"], "priority": current["priority"]},
                        },
                        "retrieved_evidence": agent_outputs["clinical_reasoning"]["citations"],
                        "citations": agent_outputs["clinical_reasoning"]["citations"],
                        "reasoning": agent_outputs["clinical_reasoning"]["explanation"],
                        "recommendation": agent_outputs["clinical_reasoning"]["recommended_action"],
                        "alert": {"decision": current["policy"], "priority": current["priority"]}, "source": "clinician_interface_demo", "mock_data": True}
            append_jsonl(AUDIT, [decision])
            feedback = {"timestamp": now, "patient_id": patient_id, "event_id": current["event_id"],
                        "features": current["vitals"], "model_version": agent_outputs["anomaly"]["model_version"],
                        "alert_policy_decision": current["policy"], "clinician_action": action,
                        "anomaly_label": None, "online_model_updated": learning_update["model_updated"],
                        "feedback": reason,
                        "source": "clinician_interface_demo", "mock_data": True}
            append_jsonl(FEEDBACK, [feedback])
            self.send_json({"ok": True, "record": decision, "learning_update": learning_update})
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[clinician-app] {fmt % args}")


if __name__ == "__main__":
    print(f"Clinician demo: http://127.0.0.1:{PORT}")
    print("Generated mock data only. P4 RAG and clinical reasoning are placeholders.")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
