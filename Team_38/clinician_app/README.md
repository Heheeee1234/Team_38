# Clinician review demo (Person 5)

This local, mock-data-only dashboard connects the generated cohort to the integrated P1-P4 workflow and the P5 observed-data alert policy. It displays patient observations, baseline and trend context, anomaly and risk outputs, experience retrieval, evidence fusion, P4 explanation and source citations, and P5 alert decisions. It does not consume Kafka messages; Kafka ingestion publishes P1-P4 handoffs to `patient.evidence` separately.

This software is for demonstration and evaluation only. It is not for clinical care, diagnosis, treatment, or deployment.

## Run

From the repository root:

```powershell
python -m pip install -e .
python clinician_app/server.py
```

Open `http://127.0.0.1:8765`. Set `CLINICIAN_APP_PORT` to use a different local port. Generate a cohort with `python -m clinical_data_gen.cli generate --output data/generated` before opening the dashboard if no generated data is present.

## P1-P5 handoff status

- **P1:** Generated observations feed the stateful trend analyzer. Kafka replay publishes context first and preserves event-time ordering.
- **P2:** The dashboard workflow maintains patient state, anomaly estimates, and clinician-reviewed experience cases. Clinician actions are not automatically treated as anomaly labels, so they do not train the online anomaly model without an explicit label.
- **P3:** Risk scoring, similar reviewed-case retrieval, and evidence fusion run for each observation.
- **P4:** Local TF-IDF retrieval and deterministic reasoning run by default. P4 receives consolidated P3 evidence and reviewed similar cases, returning a structured explanation, contributing factors, cautious review action, confidence, and citations to retrieved corpus passages. Dense embeddings and Groq tool-calling are optional.
- **P5:** The local dashboard applies the observed-data alert policy, accepts Accept, Dismiss, Defer, or Investigate actions, and records audit/feedback JSONL.

## Clinician actions and audit

Actions and optional reasons append to `data/evaluation/clinician_audit.jsonl`; a learning handoff is also written to `data/evaluation/clinician_feedback.jsonl`. The feedback is recorded in the current in-memory workflow's experience memory. It is not persisted as an online model update, and an action without an explicit anomaly label never trains the anomaly model. Audit records include observed inputs, P1-P5 outputs, policy evidence, P4 retrieved passages and citations, and clinician decisions.

## Evaluation

`/api/dashboard` reports reference metrics. To create saved reports, run:

```powershell
python scripts/evaluate_alert_policy.py
```

The policy uses patient-specific warmup baselines from up to 30 observations, requires persistent evidence for alerting, suppresses suspect-quality readings, deduplicates an active alert episode, and applies a 30-minute cooldown. Ground-truth labels are read only by the offline evaluator, not by the live workflow or policy. Reported precision, recall, lead time, alerts per patient-hour, duplicate suppression, artifact false-positive rate, citation coverage, and audit completeness are demo metrics, not clinical validation results.

## P4 options

The dashboard uses local TF-IDF retrieval and deterministic reasoning by default. Set `P4_RETRIEVAL_BACKEND=dense` and install `python -m pip install -e ".[p4-local]"` to use the local sentence-transformer vector index. To opt in to Groq tool-calling, install `python -m pip install -e ".[p4-groq]"`, provide `GROQ_API_KEY` through a secret manager or process environment, and set `P4_LIVE_LLM=true`. This sends supplied evidence to Groq Cloud; do not use real patient identifiers or protected health information.

The generator's separate online anomaly-model evaluation is in `data/evaluation/online_lr/metrics.json`; it is event-level and should not be combined with episode-level alert-policy metrics.
