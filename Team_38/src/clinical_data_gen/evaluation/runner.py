"""End-to-end prequential evaluation for the anomaly agent."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from clinical_data_gen.anomaly.anomaly_agent import AnomalyAgent
from clinical_data_gen.evaluation.online_metrics import (
    OnlineMetricsEvaluator,
)
from clinical_data_gen.state.patient_state import PatientState


class OnlineEvaluationRunner:
    """
    Run prequential evaluation over an ordered patient-state stream.

    For every observation:

        1. Predict using the current model.
        2. Record the prediction.
        3. Update the model using the supplied label.

    This ordering prevents the current label from leaking into its
    prediction.
    """

    def __init__(
        self,
        agent: AnomalyAgent,
        *,
        evaluator: OnlineMetricsEvaluator | None = None,
    ) -> None:
        self.agent = agent
        self.evaluator = (
            evaluator
            if evaluator is not None
            else OnlineMetricsEvaluator()
        )

    def evaluate(
        self,
        observations: Iterable[tuple[PatientState, int]],
    ) -> OnlineMetricsEvaluator:
        """
        Evaluate an ordered stream of (PatientState, label) pairs.

        The stream must be ordered chronologically within each patient.
        """

        for state, label in observations:
            result = self.agent.predict(state)

            loss = self.agent.update(
                state,
                label,
            )

            self.evaluator.add_prediction(
                patient_id=result.patient_id,
                timestamp=result.timestamp,
                probability=result.probability,
                true_label=label,
                loss=loss,
                model_version=result.model_version,
            )

        return self.evaluator

    def evaluate_iterator(
        self,
        observations: Iterator[tuple[PatientState, int]],
    ) -> OnlineMetricsEvaluator:
        """Evaluate a one-pass iterator without materializing it."""

        return self.evaluate(observations)

    def save_results(
        self,
        output_dir: str | Path,
    ) -> dict:
        """
        Save aggregate and rolling metrics.

        Files created:

            metrics.json
            rolling_metrics.json
            predictions.jsonl
        """

        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics = self.evaluator.compute()

        metrics_path = output_path / "metrics.json"

        metrics_path.write_text(
            json.dumps(
                metrics.as_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

        rolling_metrics = (
            self.evaluator.rolling_metrics()
        )

        rolling_path = (
            output_path / "rolling_metrics.json"
        )

        rolling_path.write_text(
            json.dumps(
                rolling_metrics,
                indent=2,
            ),
            encoding="utf-8",
        )

        predictions_path = (
            output_path / "predictions.jsonl"
        )

        with predictions_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in self.evaluator.records:
                file.write(
                    json.dumps(
                        asdict(record),
                        default=str,
                    )
                    + "\n"
                )

        return {
            "metrics": str(metrics_path),
            "rolling_metrics": str(rolling_path),
            "predictions": str(predictions_path),
        }