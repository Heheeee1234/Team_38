"""Run the deterministic P4 RAG and reasoning path on generated evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_reasoning.adapters import evidence_from_p3
from clinical_reasoning.pipeline import P4Service


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--patient-id", default="SYN-0002")
    parser.add_argument("--corpus", type=Path, default=ROOT / "knowledge" / "protocol_corpus.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "derived" / "p4_vector_index.sqlite3")
    parser.add_argument("--retrieval", choices=("dense", "tfidf"), default="dense")
    parser.add_argument("--groq", action="store_true", help="Use Groq tool calling (requires GROQ_API_KEY).")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    args = parser.parse_args()

    data_dir = args.data_dir
    corpus = args.corpus
    contexts = {row["patient_id"]: row.get("payload", row) for row in _read_jsonl(data_dir / "patient_context.jsonl")}
    observations = [row for row in _read_jsonl(data_dir / "vitals_observed.jsonl") if row.get("patient_id") == args.patient_id]
    if not observations:
        parser.error(f"No generated observations found for {args.patient_id} in {data_dir}")
    latest = observations[-1]
    payload = latest.get("payload", latest)
    values = payload.get("vitals", payload.get("values", {}))
    timestamp = latest.get("event_time") or latest.get("timestamp") or latest.get("observed_at") or "unknown"
    evidence = evidence_from_p3(
        evidence_id=f"{args.patient_id}:{timestamp}",
        patient_id=args.patient_id,
        window_end=str(timestamp),
        current_vitals=values,
        trend_result=None,
        anomaly_result=None,
        risk_result=None,
        patient_context_payload=contexts.get(args.patient_id, {}),
        conflicts_or_missing=["Trend, anomaly, risk, and fusion outputs were not supplied; this demo only exercises P4 RAG and reasoning."],
    )
    output = P4Service(
        corpus, live_llm=args.groq, retrieval_backend=args.retrieval,
        index_path=args.index, model=args.model,
    ).reason(evidence)
    print(json.dumps(output.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
