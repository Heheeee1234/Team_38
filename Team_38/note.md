## demo: run the clinician dashboard

We already have generated demo data, so **you do not need to regenerate the cohort**. From the repository root, open PowerShell and run:

```powershell
python -m pip install -e .
$env:P4_RETRIEVAL_BACKEND = "tfidf"
$env:P4_LIVE_LLM = "false"
python clinician_app\server.py
```

Then open **http://127.0.0.1:8765** in your browser. Keep that PowerShell window open while recording. If port 8765 is taken, use `$env:CLINICIAN_APP_PORT = "8766"` before starting and open port 8766 instead.

The dashboard processes the generated cohort through P1–P4 locally. P4 uses TF-IDF retrieval and deterministic reasoning by default, so you don’t need a Groq key or an embedding-model download. The demo is synthetic and not for clinical use.

## Suggested video walkthrough

1. **Introduce the flow:** synthetic observations → P1 trend → P2 anomaly/learning → P3 risk, fusion, and experience → P4 retrieval/reasoning → P5 alert review and audit.
2. **Select a patient** from the queue. The list prioritizes higher-priority cases; refresh if needed.
3. **Show the evidence panels:** patient context and vitals, then trend, anomaly, risk, experience memory, fusion, and the P5 policy decision.
4. **Show P4:** its explanation, contributing factors, suggested action for clinician review, confidence, and retrieved protocol citations.
5. **Optionally record a clinician action** such as Investigate and show it in the audit section. Actions append to the local audit and feedback JSONL files.

A useful narration note: P4 provides prototype decision support from the synthetic evidence and corpus; it is not a diagnosis or treatment recommendation.

## Kafka demo

Kafka replay is separate from the dashboard—the dashboard reads the generated cohort directly and doesn’t consume `patient.evidence`. If you already have a Kafka broker running, use two PowerShell windows:

```powershell
# Window 1
$env:KAFKA_BROKER = "localhost:9092"
$env:P4_RETRIEVAL_BACKEND = "tfidf"
$env:P4_LIVE_LLM = "false"
python -m ingestion.consumer
```

```powershell
# Window 2
python -m ingestion.producer --input data\generated\vitals_observed.jsonl --speed 60
```

The producer publishes patient context before the time-ordered vital events; the consumer publishes P1–P4 outputs and citations to `patient.evidence`. This repository doesn’t include a Kafka broker launcher, so use the dashboard walkthrough if you don’t already have a broker available.

See the `main run guide`, `clinician demo notes`, and `P4 setup/details` for more information.