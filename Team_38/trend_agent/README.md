# Trend Agent

## Purpose

The Trend Agent is the stateful trend-analysis component as proposed in the
architecture for the Agentic Clinical Deterioration and Escalation Copilot. It
consumes the observed-vitals event stream, keeps a short independent history for
each patient, and reports sustained directional changes across multiple vital
signals. Its output is intended for downstream anomaly, risk, evidence-fusion,
and escalation components.

This folder implements the Trend Analyzer only. It does not diagnose, calculate
NEWS2/qSOFA, retrieve guideline text, make treatment recommendations, or expose
clinician decision controls. This is a synthetic-data software demonstration,
not clinically validated software and not for patient care.

## Setup and run

Requires Python 3.10 or later. The implementation uses only the Python standard
library; no package installation is required.

From the project root, run:

```powershell
python trend_agent/cli.py
```

The default command reads `trend_agent/mock/patient_context.jsonl` and
`trend_agent/mock/vitals_observed.jsonl`, then writes one analysis record per
vital event to `trend_agent/mock/trend_analysis.jsonl`.

To select other files:

```powershell
python trend_agent/cli.py `
  --context path/to/patient_context.jsonl `
  --input path/to/vitals_observed.jsonl `
  --output path/to/trend_analysis.jsonl `
  --window-size 6
```

The caller should provide each patient's context before replaying that patient's
vitals. Each invocation starts with empty trend history; use one process for a
continuous replay. Context is recognized and associated with patient IDs, while
the context values themselves remain available for later copilot components.

## Input format

Both files are UTF-8 JSON Lines (one JSON object per line). Events follow the
project's common envelope: `schema_version`, `event_id`, `event_time`,
`patient_id`, `run_id`, `source`, and `payload`.

### Static context

The mock context record has a `patient_id` and a payload containing age, sex,
relevant history, current medications, and recent labs. These match the report's
static patient context contract. The Trend Agent associates context by patient
ID but does not use it to alter thresholds in this initial implementation.

### Observed vitals

The live event's `payload.vitals` contains available numeric values among:
`heart_rate`, `spo2`, `respiratory_rate`, `systolic_bp`, and `diastolic_bp`.
Additional fields such as `map` and `fio2` are accepted in the event and ignored
by this analyzer. `sequence` and `quality` may also appear in the payload. As
required by the report's leakage boundary, live events must not contain hidden
scenario labels, risk labels, episode IDs, or latent values. The included mock
has a single transient abnormal reading followed by a multi-reading adverse
trajectory; it is a hand-authored format example, not a clinical dataset.

## Output format

Each input vital event yields one JSONL analysis object containing:

- `patient_id`, `event_time`, and `source_event_id` for traceability.
- `window_points`, `ready`, and `trends` for the current rolling window.
- Per-vital direction, first/latest value, change, directional consistency,
  sample count, and the configured adverse-change threshold.
- `concurrent_worsening_signals`, a human-readable explanation, and
  `candidate_deterioration_trend`.
- `clinical_context_available` and a non-diagnostic disclaimer.

The initial thresholds are transparent demonstration parameters: heart rate
rising by 10 bpm, SpO2 falling by 3 percentage points, respiratory rate rising
by 4/min, systolic BP falling by 10 mmHg, and diastolic BP falling by 8 mmHg.
The defaults use a six-reading window, require at least four readings before a
candidate can be raised, and require at least two adverse signals. A signal also
needs at least 60% directional consistency in its available window samples.
Systolic and diastolic pressure are reported separately but count as one
physiological domain when deciding whether changes are multi-parameter.
These are not validated clinical thresholds and must be calibrated against the
intended evaluation protocol before integration.

## Dependencies and configuration

- Python 3.10+
- Python standard library only
- CLI options: `--context`, `--input`, `--output`, `--window-size`
- `TrendAgent(window_size=6, min_points=4, required_signals=2)` can also be
  imported and connected directly to an event consumer.

Input events are expected in event-time order per patient. This demo does not
persist state across restarts, deduplicate event IDs, or enforce time-based
window expiry. Those behaviors belong in the production stream/state manager.

## Architecture fit

```text
patient.context ──> patient context available by patient_id
                         │
vitals.raw ──> Trend Agent (bounded per-patient history)
                         │
                         └──> per-event trend analysis JSONL
                                  └──> downstream anomaly/risk/evidence fusion
```

