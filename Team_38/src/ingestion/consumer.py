import json
import os
import signal
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import (
    Consumer,
    Producer,
    KafkaError,
)

from clinical_data_gen.pipeline import DeteriorationWorkflow
from .models import VitalEvent, validate_context_event

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")

INPUT_TOPIC = "vitals.raw"
CONTEXT_TOPIC = "patient.context"
EVIDENCE_TOPIC = "patient.evidence"
DLQ_TOPIC = "vitals.dlq"

GROUP_ID = "clinical-ingestion-v1"

running = True
workflow = DeteriorationWorkflow()


def shutdown_handler(signum, frame):

    global running

    print("\n[CONSUMER] Shutdown requested.")

    running = False


signal.signal(
    signal.SIGINT,
    shutdown_handler,
)

signal.signal(
    signal.SIGTERM,
    shutdown_handler,
)


# =========================================================
# Kafka
# =========================================================


def create_consumer() -> Consumer:

    return Consumer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "group.id": GROUP_ID,
            # We commit manually after processing.
            "enable.auto.commit": False,
            # Useful for local development.
            "auto.offset.reset": "earliest",
            # Maximum time between poll calls.
            "max.poll.interval.ms": 300000,
        }
    )


def create_producer() -> Producer:

    return Producer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "acks": "all",
            "compression.type": "snappy",
        }
    )


def publish_confirmed(
    producer: Producer,
    *,
    topic: str,
    key: str,
    value: bytes,
) -> None:
    delivery_errors = []

    def on_delivery(error: Any, _message: Any) -> None:
        if error is not None:
            delivery_errors.append(str(error))

    producer.produce(topic=topic, key=key, value=value, callback=on_delivery)
    producer.poll(0)
    remaining = producer.flush(timeout=10)
    if remaining or delivery_errors:
        raise RuntimeError(
            f"Kafka delivery failed for topic {topic}: "
            f"pending={remaining}, errors={delivery_errors}"
        )


# =========================================================
# Validation
# =========================================================


def validate_ranges(
    event: VitalEvent,
) -> list[str]:

    errors = []

    # These are DATA QUALITY limits.
    #
    # They are NOT clinical risk thresholds.

    if event.heart_rate is not None:

        if event.heart_rate <= 0 or event.heart_rate > 300:

            errors.append(f"invalid heart_rate=" f"{event.heart_rate}")

    if event.spo2 is not None:

        if event.spo2 < 0 or event.spo2 > 100:

            errors.append(f"invalid spo2=" f"{event.spo2}")

    if event.respiratory_rate is not None:

        if event.respiratory_rate <= 0 or event.respiratory_rate > 150:

            errors.append(f"invalid respiratory_rate=" f"{event.respiratory_rate}")

    if event.systolic_bp is not None:

        if event.systolic_bp <= 0 or event.systolic_bp > 300:

            errors.append(f"invalid systolic_bp=" f"{event.systolic_bp}")

    if event.diastolic_bp is not None:

        if event.diastolic_bp <= 0 or event.diastolic_bp > 200:

            errors.append(f"invalid diastolic_bp=" f"{event.diastolic_bp}")

    if event.temperature is not None and not 25 <= event.temperature <= 45:
        errors.append(f"invalid temperature={event.temperature}")

    if (
        event.mean_arterial_pressure is not None
        and not 0 < event.mean_arterial_pressure <= 300
    ):
        errors.append(f"invalid map={event.mean_arterial_pressure}")

    if event.fio2 is not None and not 0.21 <= event.fio2 <= 1:
        errors.append(f"invalid fio2={event.fio2}")

    return errors


def validate_timestamp(
    event: VitalEvent,
) -> list[str]:

    errors = []

    now = datetime.now(timezone.utc)

    future_seconds = (event.event_time - now).total_seconds()

    # Reject events more than 5 minutes
    # into the future.
    if future_seconds > 300:

        errors.append("event_time is too far in the future")

    return errors


# =========================================================
# Normalization
# =========================================================


def normalize_event(
    event: VitalEvent,
) -> VitalEvent:

    # Event parsing normalizes naive timestamps to UTC.
    #
    # Keep this function as the boundary where
    # future source-specific normalization can go.

    return event


# =========================================================
# Quality Control
# =========================================================


def quality_control(
    event: VitalEvent,
) -> list[str]:

    flags = []

    # Missing values are not automatically invalid.
    #
    # A monitor may legitimately omit a channel.
    # Downstream components can decide how to handle
    # missing observations.

    if event.heart_rate is None:
        flags.append("MISSING_HEART_RATE")

    if event.spo2 is None:
        flags.append("MISSING_SPO2")

    if event.respiratory_rate is None:
        flags.append("MISSING_RESPIRATORY_RATE")

    if event.systolic_bp is None:
        flags.append("MISSING_SYSTOLIC_BP")

    if event.diastolic_bp is None:
        flags.append("MISSING_DIASTOLIC_BP")

    return flags


# =========================================================
# DLQ
# =========================================================


def publish_dlq(
    producer: Producer,
    raw_event,
    errors: list[str],
):

    payload = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "raw_event": raw_event,
    }

    patient_id = "unknown"

    if isinstance(raw_event, dict):

        patient_id = str(
            raw_event.get(
                "patient_id",
                "unknown",
            )
        )

    publish_confirmed(
        producer,
        topic=DLQ_TOPIC,
        key=patient_id,
        value=json.dumps(payload).encode("utf-8"),
    )


# =========================================================
# Downstream interface
# =========================================================


def process_valid_event(
    event: VitalEvent,
    evidence_producer: Producer,
):
    observed_event = event.model_dump(mode="json")
    agent_outputs = workflow.process_event(observed_event)
    handoff = {
        "event": observed_event,
        "agent_outputs": agent_outputs,
        "person4_clinical_reasoning": agent_outputs["clinical_reasoning"],
    }
    publish_confirmed(
        evidence_producer,
        topic=EVIDENCE_TOPIC,
        key=event.patient_id,
        value=json.dumps(handoff).encode("utf-8"),
    )
    print(
        "[DOWNSTREAM] "
        f"patient={event.patient_id} event={event.event_id} "
        f"risk={agent_outputs['risk']['risk_level']} "
        f"trend_candidate={agent_outputs['trend']['candidate_deterioration_trend']}"
    )


# =========================================================
# Main consumer loop
# =========================================================


def run():

    consumer = create_consumer()

    output_producer = create_producer()

    consumer.subscribe([INPUT_TOPIC, CONTEXT_TOPIC])

    print(f"[CONSUMER] Broker: " f"{KAFKA_BROKER}")

    print(f"[CONSUMER] Input: " f"{INPUT_TOPIC}")

    print(f"[CONSUMER] Context: " f"{CONTEXT_TOPIC}")

    print(f"[CONSUMER] Evidence output: " f"{EVIDENCE_TOPIC}")

    print(f"[CONSUMER] DLQ: " f"{DLQ_TOPIC}")

    print(
        "[CONSUMER] P4 retrieval: "
        f"{os.environ.get('P4_RETRIEVAL_BACKEND', 'tfidf')}; "
        f"live LLM: {os.environ.get('P4_LIVE_LLM', 'false')}"
    )

    try:

        while running:

            message = consumer.poll(timeout=1.0)

            if message is None:
                continue

            if message.error():

                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue

                print(f"[KAFKA ERROR] " f"{message.error()}")

                continue

            # =================================================
            # 1. Deserialize
            # =================================================

            raw_value = message.value() or b""
            try:
                raw_event = json.loads(raw_value.decode("utf-8"))

            except (UnicodeDecodeError, json.JSONDecodeError) as exc:

                errors = [f"invalid JSON: {exc}"]

                print("[INVALID JSON] " f"{errors}")

                publish_dlq(
                    producer=output_producer,
                    raw_event={
                        "raw": raw_value.decode("utf-8", errors="replace")
                    },
                    errors=errors,
                )

                # Poison message has been handled.
                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                continue

            if not isinstance(raw_event, dict):
                errors = ["event must be a JSON object"]
                publish_dlq(output_producer, {"raw": raw_event}, errors)
                consumer.commit(message=message, asynchronous=False)
                continue

            # Patient context is consumed before vital replay and registered
            # with the same stateful workflow used for vital events.
            if message.topic() == CONTEXT_TOPIC:
                try:
                    workflow.process_context(validate_context_event(raw_event))
                except (TypeError, ValueError) as exc:
                    errors = [f"patient context validation failed: {exc}"]
                    publish_dlq(output_producer, raw_event, errors)
                    consumer.commit(message=message, asynchronous=False)
                    continue
                consumer.commit(message=message, asynchronous=False)
                continue

            # =================================================
            # 2. Schema validation
            # =================================================

            try:

                event = VitalEvent.model_validate(raw_event)

            except (TypeError, ValueError, KeyError) as exc:

                errors = [
                    "schema validation failed",
                    str(exc),
                ]

                print(
                    "[SCHEMA INVALID] "
                    f"patient="
                    f"{raw_event.get('patient_id', 'unknown')} "
                    f"errors={errors}"
                )

                publish_dlq(
                    producer=output_producer,
                    raw_event=raw_event,
                    errors=errors,
                )

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                continue

            # =================================================
            # 3. Range validation
            # =================================================

            errors = validate_ranges(event)

            # =================================================
            # 4. Timestamp validation
            # =================================================

            errors.extend(validate_timestamp(event))

            if errors:

                print(
                    "[VALIDATION FAILED] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"errors={errors}"
                )

                publish_dlq(
                    producer=output_producer,
                    raw_event=raw_event,
                    errors=errors,
                )

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                continue

            # =================================================
            # 5. Normalize
            # =================================================

            event = normalize_event(event)

            # =================================================
            # 6. Quality control
            # =================================================

            quality_flags = quality_control(event)

            if quality_flags:

                print(
                    "[QUALITY FLAGS] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"flags={quality_flags}"
                )

            # =================================================
            # 7. Downstream handoff
            # =================================================

            try:

                process_valid_event(event, output_producer)

            except Exception as exc:

                # IMPORTANT:
                # Do NOT commit if downstream processing
                # fails.
                #
                # Kafka can then redeliver the message.

                print(
                    "[DOWNSTREAM ERROR] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"error={exc}"
                )

                continue

            # =================================================
            # 8. Commit after successful processing
            # =================================================

            consumer.commit(
                message=message,
                asynchronous=False,
            )

    finally:

        print("[CONSUMER] Flushing output producer...")

        output_producer.flush()

        print("[CONSUMER] Closing consumer...")

        consumer.close()

        print("[CONSUMER] Shutdown complete.")


if __name__ == "__main__":
    run()
