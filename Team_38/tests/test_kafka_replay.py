import json

import pytest

from ingestion.producer import replay_events


class FakeProducer:
    def __init__(self):
        self.messages = []
        self._topic = None
        self._key = None

    def produce(self, *, topic, key, value, callback):
        self._topic = topic
        self._key = key.encode("utf-8")
        self.messages.append((topic, key, json.loads(value)))
        callback(None, self)

    def poll(self, timeout):
        pass

    def flush(self, timeout=None):
        return 0

    def topic(self):
        return self._topic

    def key(self):
        return self._key

    def partition(self):
        return 0

    def offset(self):
        return len(self.messages) - 1


def _event(event_id, patient_id, event_time):
    return {
        "schema_version": "1.0",
        "event_id": event_id,
        "event_time": event_time,
        "patient_id": patient_id,
        "run_id": "test-run",
        "source": "test",
        "payload": {
            "vitals": {
                "heart_rate": 80,
                "spo2": 97,
                "respiratory_rate": 18,
            },
            "quality": "normal",
        },
    }


def test_replay_publishes_context_then_event_time_order_and_spacing():
    producer = FakeProducer()
    delays = []
    contexts = [{
        "schema_version": "1.0",
        "event_id": "context-p1",
        "event_time": "2026-01-01T00:00:00Z",
        "patient_id": "p1",
        "run_id": "test-run",
        "source": "test",
        "payload": {"age": 65},
    }]
    events = [
        _event("later", "p1", "2026-01-01T00:01:00Z"),
        _event("earlier", "p1", "2026-01-01T00:00:00Z"),
    ]

    replay_events(producer, contexts, events, speed=60, sleep=delays.append)

    assert [topic for topic, _, _ in producer.messages] == [
        "patient.context",
        "vitals.raw",
        "vitals.raw",
    ]
    assert [message[2]["event_id"] for message in producer.messages[1:]] == [
        "earlier",
        "later",
    ]
    assert delays == [1.0]


def test_replay_rejects_nonpositive_speed():
    with pytest.raises(ValueError, match="greater than zero"):
        replay_events(FakeProducer(), [], [], speed=0)


def test_replay_rejects_invalid_event_schema():
    with pytest.raises(ValueError, match="event_time"):
        replay_events(FakeProducer(), [], [{"event_time": "invalid"}], speed=1)


def test_replay_rejects_incomplete_context_schema():
    with pytest.raises(ValueError, match="missing fields"):
        replay_events(FakeProducer(), [{"patient_id": "p1"}], [], speed=1)


def test_consumer_runs_pipeline_and_emits_explicit_p4_handoff():
    from ingestion.consumer import EVIDENCE_TOPIC, process_valid_event
    from ingestion.models import VitalEvent

    event = VitalEvent.model_validate(_event(
        "consumer-event",
        "consumer-patient",
        "2026-01-01T00:00:00Z",
    ))
    producer = FakeProducer()

    process_valid_event(event, producer)

    topic, key, handoff = producer.messages[0]
    assert topic == EVIDENCE_TOPIC
    assert key == "consumer-patient"
    assert handoff["agent_outputs"]["risk"]["risk_level"]
    assert handoff["person4_clinical_reasoning"]["model"] == "deterministic-p4"
    assert handoff["person4_clinical_reasoning"]["citations"]
