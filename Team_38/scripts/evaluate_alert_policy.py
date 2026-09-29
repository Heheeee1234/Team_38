"""Evaluate Person 5 policy against held-out episode labels.

Run from the project folder: python scripts/evaluate_alert_policy.py
This is an evaluation script, not an online alert-path dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinician_app.evaluation import calculate_metrics
from clinician_app.policy import evaluate_stream
from clinician_app.server import AUDIT, input_data, make_audit_traces, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the observed-data-only mock alert policy.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "alert_policy")
    args = parser.parse_args()
    events, contexts, episodes, labels = input_data()
    if not events:
        raise SystemExit("No generated events found. Generate a mock cohort before evaluating.")
    decisions = evaluate_stream(events)
    make_audit_traces(events, decisions, contexts)
    event_ids = {event["event_id"] for event in events}
    traces = [row for row in read_jsonl(AUDIT) if row.get("event_type") == "policy_evaluation" and row.get("event_id") in event_ids]
    metrics, predictions = calculate_metrics(decisions, episodes, labels, traces)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "episode_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    alert_rows = [decision for stream in decisions.values() for decision in stream if decision["policy"] == "ALERT"]
    with (args.output / "alert_events.jsonl").open("w", encoding="utf-8") as stream:
        for row in alert_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved evaluation outputs to {args.output}")


if __name__ == "__main__":
    main()
