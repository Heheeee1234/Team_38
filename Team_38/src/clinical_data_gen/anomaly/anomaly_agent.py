"""Anomaly agent combining patient-state features with online learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clinical_data_gen.anomaly.features import (
    FeatureVector,
    PatientFeatureExtractor,
)
from clinical_data_gen.learning.online_logistic_regression import (
    OnlineLogisticRegression,
    OnlinePrediction,
)
from clinical_data_gen.state.patient_state import PatientState


@dataclass(frozen=True)
class AnomalyResult:
    """Result produced for one patient-state observation."""

    patient_id: str
    timestamp: Any
    probability: float
    is_anomalous: bool
    feature_vector: dict[str, float]
    contributing_features: list[str]
    model_version: int


@dataclass(frozen=True)
class AnomalyFeedback:
    """Clinician feedback with anomaly correctness kept separate from action."""

    patient_id: str
    timestamp: Any
    anomaly_label: bool | None
    decision: str | None = None
    reason: str | None = None


class AnomalyAgent:
    """
    Online anomaly-learning agent.

    The lifecycle for a labelled observation is:

        result = agent.predict(state)
        agent.update(state, label)

    Prediction always happens before learning from the label.
    This prevents the current label from leaking into its prediction.
    """

    def __init__(
        self,
        *,
        feature_extractor: PatientFeatureExtractor | None = None,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        positive_class_weight: float = 1.0,
        gradient_clip: float = 5.0,
        probability_threshold: float = 0.5,
        diminishing_learning_rate: bool = True,
    ) -> None:
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else PatientFeatureExtractor()
        )

        self.model = OnlineLogisticRegression(
            n_features=self.feature_extractor.n_features,
            learning_rate=learning_rate,
            l2=l2,
            positive_class_weight=positive_class_weight,
            gradient_clip=gradient_clip,
            probability_threshold=probability_threshold,
            diminishing_learning_rate=diminishing_learning_rate,
        )

    @property
    def model_version(self) -> int:
        return self.model.model_version

    @property
    def update_count(self) -> int:
        return self.model.update_count

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_extractor.feature_names

    def predict(self, state: PatientState) -> AnomalyResult:
        """
        Predict anomaly probability for the current patient state.

        No label or future information is used here.
        """

        feature_vector = self.feature_extractor.extract(state)

        prediction = self.model.predict(feature_vector.values)

        contributing_features = self._get_contributing_features(
            feature_vector,
        )

        return AnomalyResult(
            patient_id=str(getattr(state, "patient_id", "")),
            timestamp=getattr(state, "timestamp", None),
            probability=prediction.probability,
            is_anomalous=bool(prediction.predicted_label),
            feature_vector=feature_vector.as_dict(),
            contributing_features=contributing_features,
            model_version=prediction.model_version,
        )

    def update(
        self,
        state: PatientState,
        label: int | AnomalyFeedback,
    ) -> float | None:
        """
        Update the online model using an explicit binary anomaly label.

        Returns the pre-update logistic loss.

        This method intentionally requires an explicit label. In the live
        pipeline, clinician feedback should only be converted into an
        anomaly label when the feedback semantics explicitly support that.
        """

        if isinstance(label, AnomalyFeedback):
            if label.patient_id != state.patient_id:
                raise ValueError("Feedback patient_id does not match PatientState")
            if label.decision is not None and label.decision.lower() not in {
                "accept", "dismiss", "defer", "investigate"
            }:
                raise ValueError(f"Unsupported clinician decision: {label.decision!r}")
            if label.anomaly_label is None:
                return None
            label = int(label.anomaly_label)

        feature_vector = self.feature_extractor.extract(state)

        return self.model.update(
            feature_vector.values,
            label,
        )

    def predict_and_update(
        self,
        state: PatientState,
        label: int,
    ) -> tuple[AnomalyResult, float]:
        """
        Convenience method for prequential evaluation.

        The prediction is generated first and the model is updated second.
        """

        result = self.predict(state)
        loss = self.update(state, label)

        return result, loss

    def predict_features(
        self,
        feature_vector: FeatureVector,
    ) -> OnlinePrediction:
        """Predict directly from an already extracted feature vector."""

        if len(feature_vector) != self.feature_extractor.n_features:
            raise ValueError(
                "Feature vector size does not match the agent model"
            )

        return self.model.predict(feature_vector.values)

    def _get_contributing_features(
        self,
        feature_vector: FeatureVector,
    ) -> list[str]:
        """
        Return features with the largest absolute weighted contribution.

        This is an interpretability aid, not a causal explanation.
        """

        contributions = []

        for name, value, weight in zip(
            feature_vector.names,
            feature_vector.values,
            self.model.weights,
        ):
            contribution = float(value) * float(weight)

            if contribution != 0.0:
                contributions.append(
                    (name, abs(contribution))
                )

        contributions.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            name
            for name, _ in contributions[:5]
        ]