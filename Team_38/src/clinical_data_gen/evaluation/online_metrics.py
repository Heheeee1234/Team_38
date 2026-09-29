"""Metrics and evaluation utilities for online binary prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction made before observing its label."""

    patient_id: str
    timestamp: object
    probability: float
    predicted_label: int
    true_label: int
    loss: float
    model_version: int


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineMetrics:
    """Complete summary of binary online-learning performance."""

    sample_count: int
    positive_count: int
    negative_count: int

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    negative_predictive_value: float
    f1: float

    roc_auc: float | None
    pr_auc: float | None

    brier_score: float
    log_loss: float
    expected_calibration_error: float

    cumulative_loss: float
    mean_loss: float

    confusion_matrix: ConfusionMatrix

    alerts_per_patient_hour: float | None

    def as_dict(self) -> dict:
        result = asdict(self)
        result["confusion_matrix"] = (
            self.confusion_matrix.as_dict()
        )
        return result


class OnlineMetricsEvaluator:
    """
    Collect and calculate metrics from streaming predictions.

    Predictions are recorded before their labels are used for model
    updates. This makes the evaluator suitable for prequential evaluation.
    """

    def __init__(
        self,
        *,
        probability_bins: int = 10,
        rolling_window: int = 100,
    ) -> None:
        if probability_bins <= 0:
            raise ValueError(
                "probability_bins must be positive"
            )

        if rolling_window <= 0:
            raise ValueError(
                "rolling_window must be positive"
            )

        self.probability_bins = probability_bins
        self.rolling_window = rolling_window
        self._records: list[PredictionRecord] = []

    @property
    def records(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._records)

    def add(
        self,
        *,
        patient_id: str,
        timestamp: object,
        probability: float,
        predicted_label: int,
        true_label: int,
        loss: float,
        model_version: int,
    ) -> None:
        """Add one prediction to the evaluation stream."""

        self._validate_probability(probability)
        self._validate_binary(predicted_label, "predicted_label")
        self._validate_binary(true_label, "true_label")

        self._records.append(
            PredictionRecord(
                patient_id=str(patient_id),
                timestamp=timestamp,
                probability=float(probability),
                predicted_label=int(predicted_label),
                true_label=int(true_label),
                loss=float(loss),
                model_version=int(model_version),
            )
        )

    def add_prediction(
        self,
        *,
        patient_id: str,
        timestamp: object,
        probability: float,
        true_label: int,
        loss: float,
        model_version: int,
        threshold: float = 0.5,
    ) -> None:
        """Add a prediction while deriving its binary label."""

        predicted_label = int(probability >= threshold)

        self.add(
            patient_id=patient_id,
            timestamp=timestamp,
            probability=probability,
            predicted_label=predicted_label,
            true_label=true_label,
            loss=loss,
            model_version=model_version,
        )

    def compute(self) -> OnlineMetrics:
        """Calculate all aggregate metrics."""

        if not self._records:
            raise ValueError(
                "Cannot compute metrics without predictions"
            )

        y_true = [record.true_label for record in self._records]
        y_pred = [
            record.predicted_label
            for record in self._records
        ]
        probabilities = [
            record.probability
            for record in self._records
        ]

        matrix = self._confusion_matrix(y_true, y_pred)

        sample_count = len(self._records)
        positive_count = sum(y_true)
        negative_count = sample_count - positive_count

        accuracy = (
            (matrix.true_positive + matrix.true_negative)
            / sample_count
        )

        sensitivity = self._safe_divide(
            matrix.true_positive,
            matrix.true_positive + matrix.false_negative,
        )

        specificity = self._safe_divide(
            matrix.true_negative,
            matrix.true_negative + matrix.false_positive,
        )

        precision = self._safe_divide(
            matrix.true_positive,
            matrix.true_positive + matrix.false_positive,
        )

        npv = self._safe_divide(
            matrix.true_negative,
            matrix.true_negative + matrix.false_negative,
        )

        if precision is None or sensitivity is None:
            f1 = 0.0
        else:
            f1 = self._safe_divide(
                2.0 * precision * sensitivity,
                precision + sensitivity,
            )
        balanced_accuracy = (
            (sensitivity + specificity) / 2.0
            if sensitivity is not None
            and specificity is not None
            else 0.0
        )

        losses = [record.loss for record in self._records]

        return OnlineMetrics(
            sample_count=sample_count,
            positive_count=positive_count,
            negative_count=negative_count,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            precision=precision or 0.0,
            recall=sensitivity or 0.0,
            specificity=specificity or 0.0,
            negative_predictive_value=npv or 0.0,
            f1=f1 or 0.0,
            roc_auc=self.roc_auc(),
            pr_auc=self.pr_auc(),
            brier_score=self.brier_score(),
            log_loss=self.log_loss(),
            expected_calibration_error=(
                self.expected_calibration_error()
            ),
            cumulative_loss=sum(losses),
            mean_loss=(
                sum(losses) / len(losses)
                if losses
                else 0.0
            ),
            confusion_matrix=matrix,
            alerts_per_patient_hour=(
                self.alerts_per_patient_hour()
            ),
        )

    def rolling_metrics(self) -> list[dict[str, float | int]]:
        """
        Calculate learning trajectory over the stream.

        Each point summarizes the most recent rolling window.
        """

        results: list[dict[str, float | int]] = []

        for end in range(
            self.rolling_window,
            len(self._records) + 1,
        ):
            window = self._records[
                end - self.rolling_window : end
            ]

            evaluator = self._from_records(window)
            metrics = evaluator.compute()

            results.append(
                {
                    "observation": end,
                    "window_size": len(window),
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "pr_auc": (
                        metrics.pr_auc
                        if metrics.pr_auc is not None
                        else 0.0
                    ),
                    "log_loss": metrics.log_loss,
                    "brier_score": metrics.brier_score,
                }
            )

        return results

    def roc_auc(self) -> float | None:
        """Calculate ROC-AUC using the recorded probabilities."""

        if not self._records:
            return None

        y_true = [
            record.true_label
            for record in self._records
        ]

        probabilities = [
            record.probability
            for record in self._records
        ]

        positives = sum(y_true)
        negatives = len(y_true) - positives

        if positives == 0 or negatives == 0:
            return None

        ranked = sorted(
            enumerate(probabilities),
            key=lambda item: item[1],
        )

        ranks = [0.0] * len(probabilities)

        index = 0

        while index < len(ranked):
            end = index

            while (
                end + 1 < len(ranked)
                and ranked[end + 1][1] == ranked[index][1]
            ):
                end += 1

            average_rank = (
                index + end + 2
            ) / 2.0

            for position in range(index, end + 1):
                original_index = ranked[position][0]
                ranks[original_index] = average_rank

            index = end + 1

        positive_rank_sum = sum(
            ranks[index]
            for index, label in enumerate(y_true)
            if label == 1
        )

        return (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        ) / (positives * negatives)

    def pr_auc(self) -> float | None:
        """Calculate area under the precision-recall curve."""

        if not self._records:
            return None

        records = sorted(
            self._records,
            key=lambda record: record.probability,
            reverse=True,
        )

        total_positives = sum(
            record.true_label
            for record in records
        )

        if total_positives == 0:
            return None

        true_positives = 0
        false_positives = 0

        previous_recall = 0.0
        area = 0.0

        index = 0

        while index < len(records):
            threshold = records[index].probability
            end = index

            while (
                end + 1 < len(records)
                and records[end + 1].probability
                == threshold
            ):
                end += 1

            for position in range(index, end + 1):
                if records[position].true_label == 1:
                    true_positives += 1
                else:
                    false_positives += 1

            recall = true_positives / total_positives
            precision = self._safe_divide(
                true_positives,
                true_positives + false_positives,
            ) or 0.0

            area += (
                recall - previous_recall
            ) * precision

            previous_recall = recall
            index = end + 1

        return area

    def brier_score(self) -> float:
        """Calculate mean squared probability error."""

        if not self._records:
            return 0.0

        return sum(
            (
                record.probability
                - record.true_label
            )
            ** 2
            for record in self._records
        ) / len(self._records)

    def log_loss(self) -> float:
        """Calculate mean binary cross-entropy."""

        if not self._records:
            return 0.0

        return sum(
            self._binary_log_loss(
                record.probability,
                record.true_label,
            )
            for record in self._records
        ) / len(self._records)

    def expected_calibration_error(self) -> float:
        """Calculate equal-width expected calibration error."""

        if not self._records:
            return 0.0

        bins: list[list[PredictionRecord]] = [
            []
            for _ in range(self.probability_bins)
        ]

        for record in self._records:
            index = min(
                int(
                    record.probability
                    * self.probability_bins
                ),
                self.probability_bins - 1,
            )

            bins[index].append(record)

        total = len(self._records)
        error = 0.0

        for bucket in bins:
            if not bucket:
                continue

            confidence = sum(
                record.probability
                for record in bucket
            ) / len(bucket)

            observed_frequency = sum(
                record.true_label
                for record in bucket
            ) / len(bucket)

            error += (
                len(bucket) / total
            ) * abs(
                confidence - observed_frequency
            )

        return error

    def alerts_per_patient_hour(self) -> float | None:
        """
        Calculate alert rate from timestamped observations.

        Timestamps may be numeric hours or objects that expose a
        ``total_seconds`` difference through subtraction.
        """

        if not self._records:
            return None

        alerts = sum(
            record.predicted_label
            for record in self._records
        )

        patient_times: dict[str, list[object]] = {}

        for record in self._records:
            patient_times.setdefault(
                record.patient_id,
                [],
            ).append(record.timestamp)

        total_hours = 0.0

        for timestamps in patient_times.values():
            duration = self._duration_hours(timestamps)

            if duration is not None:
                total_hours += max(duration, 0.0)

        if total_hours <= 0.0:
            return None

        return alerts / total_hours

    @staticmethod
    def _duration_hours(
        timestamps: Sequence[object],
    ) -> float | None:
        if len(timestamps) < 2:
            return None

        try:
            start = timestamps[0]
            end = timestamps[-1]
            difference = end - start

            if hasattr(difference, "total_seconds"):
                return (
                    difference.total_seconds() / 3600.0
                )

            return float(difference) / 3600.0

        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _confusion_matrix(
        y_true: Sequence[int],
        y_pred: Sequence[int],
    ) -> ConfusionMatrix:
        tn = fp = fn = tp = 0

        for actual, predicted in zip(
            y_true,
            y_pred,
        ):
            if actual == 1 and predicted == 1:
                tp += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1
            else:
                tn += 1

        return ConfusionMatrix(
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            true_positive=tp,
        )

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
    ) -> float | None:
        if denominator == 0:
            return None

        return numerator / denominator

    @staticmethod
    def _binary_log_loss(
        probability: float,
        label: int,
    ) -> float:
        epsilon = 1e-15

        probability = min(
            max(probability, epsilon),
            1.0 - epsilon,
        )

        if label == 1:
            from math import log

            return -log(probability)

        from math import log

        return -log(1.0 - probability)

    @staticmethod
    def _validate_probability(
        probability: float,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

    @staticmethod
    def _validate_binary(
        value: int,
        name: str,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be 0 or 1, not bool"
            )

        if value not in (0, 1):
            raise ValueError(
                f"{name} must be either 0 or 1"
            )

    @classmethod
    def _from_records(
        cls,
        records: Iterable[PredictionRecord],
    ) -> "OnlineMetricsEvaluator":
        evaluator = cls()

        evaluator._records = list(records)

        return evaluator

    def to_dict(self) -> dict:
        """Return the complete evaluation result as JSON-ready data."""

        return self.compute().as_dict()