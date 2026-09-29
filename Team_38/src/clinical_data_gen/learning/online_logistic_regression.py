"""Online logistic regression for streaming binary prediction."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence


@dataclass(frozen=True)
class OnlinePrediction:
    """Prediction returned by the online logistic regression model."""

    probability: float
    predicted_label: int
    model_version: int


class OnlineLogisticRegression:
    """
    Incremental binary logistic regression.

    The model processes one observation at a time:

        predict(x)
        update(x, y)

    The prediction must happen before the update so that the current
    ground-truth label cannot leak into the current prediction.
    """

    def __init__(
        self,
        n_features: int,
        *,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        positive_class_weight: float = 1.0,
        gradient_clip: float = 5.0,
        probability_threshold: float = 0.5,
        diminishing_learning_rate: bool = True,
    ) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if l2 < 0:
            raise ValueError("l2 must be non-negative")

        if positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive")

        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

        if not 0.0 < probability_threshold < 1.0:
            raise ValueError("probability_threshold must be in (0, 1)")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.positive_class_weight = positive_class_weight
        self.gradient_clip = gradient_clip
        self.probability_threshold = probability_threshold
        self.diminishing_learning_rate = diminishing_learning_rate

        self._weights = [0.0] * n_features
        self._bias = 0.0

        self._updates = 0
        self._model_version = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> int:
        """Return the number of completed parameter updates."""
        return self._model_version

    @property
    def update_count(self) -> int:
        """Return the number of training observations processed."""
        return self._updates

    @property
    def weights(self) -> tuple[float, ...]:
        """Return a read-only snapshot of model weights."""
        return tuple(self._weights)

    @property
    def bias(self) -> float:
        """Return the current model bias."""
        return self._bias

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: Sequence[float]) -> float:
        """
        Predict P(y=1 | x).

        No model parameters are modified by this operation.
        """
        self._validate_features(features)

        score = self._bias

        for weight, feature in zip(self._weights, features):
            score += weight * float(feature)

        return self._sigmoid(score)

    def predict(self, features: Sequence[float]) -> OnlinePrediction:
        """Return probability, binary prediction and model version."""
        probability = self.predict_proba(features)

        return OnlinePrediction(
            probability=probability,
            predicted_label=int(
                probability >= self.probability_threshold
            ),
            model_version=self._model_version,
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        features: Sequence[float],
        label: int,
    ) -> float:
        """
        Perform one online gradient update.

        Returns:
            The logistic loss for this observation before the update.

        The caller should always follow:

            prediction = model.predict(x)
            loss = model.update(x, y)

        rather than updating before prediction.
        """
        self._validate_features(features)
        self._validate_label(label)

        probability = self.predict_proba(features)

        loss = self._log_loss(probability, label)

        error = probability - float(label)

        sample_weight = (
            self.positive_class_weight
            if label == 1
            else 1.0
        )

        weighted_error = sample_weight * error

        learning_rate = self._current_learning_rate()

        # Gradient clipping protects the online learner from a single
        # pathological observation dominating the parameter update.
        weighted_error = self._clip(
            weighted_error,
            -self.gradient_clip,
            self.gradient_clip,
        )

        for index, feature in enumerate(features):
            gradient = (
                weighted_error * float(feature)
                + self.l2 * self._weights[index]
            )

            gradient = self._clip(
                gradient,
                -self.gradient_clip,
                self.gradient_clip,
            )

            self._weights[index] -= learning_rate * gradient

        bias_gradient = self._clip(
            weighted_error,
            -self.gradient_clip,
            self.gradient_clip,
        )

        self._bias -= learning_rate * bias_gradient

        self._updates += 1
        self._model_version += 1

        return loss

    # ------------------------------------------------------------------
    # Learning-rate schedule
    # ------------------------------------------------------------------

    def _current_learning_rate(self) -> float:
        if not self.diminishing_learning_rate:
            return self.learning_rate

        # Standard diminishing schedule:
        #
        #     eta_t = eta_0 / sqrt(t)
        #
        # This is appropriate for online convex optimization and avoids
        # taking equally large steps forever.
        return self.learning_rate / max(1.0, self._updates + 1) ** 0.5

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_features(self, features: Sequence[float]) -> None:
        if len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, "
                f"received {len(features)}"
            )

        for index, value in enumerate(features):
            if isinstance(value, bool):
                raise TypeError(
                    f"Feature {index} cannot be boolean"
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Feature {index} must be numeric"
                ) from exc

            if not self._is_finite(numeric_value):
                raise ValueError(
                    f"Feature {index} must be finite"
                )

    @staticmethod
    def _validate_label(label: int) -> None:
        if isinstance(label, bool):
            raise TypeError("label must be 0 or 1, not bool")

        if label not in (0, 1):
            raise ValueError("label must be either 0 or 1")

    # ------------------------------------------------------------------
    # Numerical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Numerically stable sigmoid."""
        if value >= 0:
            z = exp(-value)
            return 1.0 / (1.0 + z)

        z = exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _log_loss(probability: float, label: int) -> float:
        """Binary log loss with probability clipping."""
        eps = 1e-15

        probability = min(
            max(probability, eps),
            1.0 - eps,
        )

        if label == 1:
            return -__import__("math").log(probability)

        return -__import__("math").log(1.0 - probability)

    @staticmethod
    def _clip(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(value, upper))

    @staticmethod
    def _is_finite(value: float) -> bool:
        return value == value and abs(value) != float("inf")