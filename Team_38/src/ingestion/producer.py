"""Replay generated patient context and vital events into Kafka."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from confluent_kafka import Producer

from ingestion.models import VitalEvent, validate_context_event

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
VITALS_TOPIC = "vitals.raw"
CONTEXT_TOPIC = "patient.context"


def create_producer(brokers: str = KAFKA_BROKER) -> Producer:
    return Producer({
        "bootstrap.servers": brokers,
        "acks": "all",
        "linger.ms": 5,
        "compression.type": "snappy",
        "message.send.max.retries": 5,
    })


def delivery_report(error: Any, message: Any) -> None:
    if error is not None:
        print(
            f"[PRODUCER ERROR] topic={message.topic()} "
            f"partition={message.partition()} error={error}"
        )
        return
    key = message.key()
    patient_id = key.decode("utf-8") if isinstance(key, bytes) else str(key)
    print(
        f"[PRODUCED] patient={patient_id} "
        f"partition={message.partition()} offset={message.offset()}"
    )


def publish_event(
    producer: Producer,
    topic: str,
    event: Mapping[str, Any],
    *,
    callback: Callable[[Any, Any], None] = delivery_report,
) -> None:
    producer.produce(
        topic=topic,
        key=str(event["patient_id"]),
        value=json.dumps(event).encode("utf-8"),
        callback=callback,
    )
    producer.poll(0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}, line {line_number}")
            records.append(record)
    return records


def _event_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("event_time must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid event_time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def replay_events(
    producer: Producer,
    contexts: Sequence[Mapping[str, Any]],
    vitals: Sequence[Mapping[str, Any]],
    *,
    speed: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if speed <= 0:
        raise ValueError("Replay speed must be greater than zero")
    ordered_vitals = sorted(vitals, key=lambda event: _event_time(event.get("event_time")))

    delivery_errors = []

    def report_delivery(error: Any, message: Any) -> None:
        delivery_report(error, message)
        if error is not None:
            delivery_errors.append(str(error))

    def flush_or_raise(stage: str) -> None:
        if producer.flush(timeout=10) or delivery_errors:
            raise RuntimeError(f"Kafka {stage} delivery failed: {delivery_errors}")

    for context in contexts:
        validated_context = validate_context_event(context)
        publish_event(producer, CONTEXT_TOPIC, validated_context, callback=report_delivery)
    flush_or_raise("patient context")

    previous_time = None
    for raw_event in ordered_vitals:
        event = VitalEvent.model_validate(raw_event).model_dump(mode="json")
        current_time = _event_time(event["event_time"])
        if previous_time is not None:
            elapsed = (current_time - previous_time).total_seconds()
            if elapsed > 0:
                sleep(elapsed / speed)
        publish_event(producer, VITALS_TOPIC, event, callback=report_delivery)
        previous_time = current_time
    flush_or_raise("vital event")


def replay_jsonl(
    input_file: str,
    speed: float = 60.0,
    context_file: str | None = None,
    brokers: str = KAFKA_BROKER,
) -> None:
    if speed <= 0:
        raise ValueError("Replay speed must be greater than zero")
    vitals_path = Path(input_file)
    if not vitals_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {vitals_path}")
    contexts_path = (
        Path(context_file)
        if context_file
        else vitals_path.with_name("patient_context.jsonl")
    )
    contexts = read_jsonl(contexts_path) if contexts_path.is_file() else []
    vitals = read_jsonl(vitals_path)
    producer = create_producer(brokers)
    print(f"[PRODUCER] Broker: {brokers}")
    print(f"[PRODUCER] Contexts: {len(contexts)}; vital events: {len(vitals)}; speed: {speed}x")
    replay_events(producer, contexts, vitals, speed=speed)
    print("[PRODUCER] Replay complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay generated events into Kafka.")
    parser.add_argument("--input", required=True, help="Path to vitals_observed.jsonl")
    parser.add_argument("--context", help="Optional path to patient_context.jsonl")
    parser.add_argument("--speed", type=float, default=60.0, help="Replay speed multiplier")
    parser.add_argument("--brokers", default=KAFKA_BROKER, help="Kafka bootstrap servers")
    args = parser.parse_args()
    replay_jsonl(
        input_file=args.input,
        context_file=args.context,
        speed=args.speed,
        brokers=args.brokers,
    )
