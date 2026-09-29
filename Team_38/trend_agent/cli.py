"""JSONL command line runner for the Trend Agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from trend_agent.trend_agent import TrendAgent


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze observed patient vital trends")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "mock" / "vitals_observed.jsonl")
    parser.add_argument("--context", type=Path, default=Path(__file__).parent / "mock" / "patient_context.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "mock" / "trend_analysis.jsonl")
    parser.add_argument("--window-size", type=int, default=6)
    args = parser.parse_args()
    agent = TrendAgent(window_size=args.window_size, min_points=min(4, args.window_size))
    for context in read_jsonl(args.context):
        agent.observe_context(context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for event in read_jsonl(args.input):
            output.write(json.dumps(agent.observe(event), sort_keys=True) + "\n")
    print(f"Wrote trend analyses to {args.output}")


if __name__ == "__main__":
    main()
