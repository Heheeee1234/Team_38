"""Feature extraction for the online anomaly-learning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from clinical_data_gen.preprocessing.constants import CORE_VITALS
from clinical_data_gen.state.patient_state import PatientState


@dataclass(frozen=True)
class FeatureVector:
    """Named feature vector consumed by the online learning model."""

    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))

    def __len__(self) -> int:
        return len(self.values)


# Fixed numerical scales keep heterogeneous physiological measurements
# in a comparable range without fitting anything on the evaluation stream.
_DEFAULT_SCALES: Mapping[str, tuple[float, float]] = {
    "HR": (80.0, 30.0),
    "O2Sat": (97.0, 5.0),
    "Temp": (37.0, 1.0),
    "SBP": (120.0, 30.0),
    "MAP": (80.0, 20.0),
    "DBP": (70.0, 20.0),
    "Resp": (18.0, 8.0),
}

_AGE_SCALE = (60.0, 20.0)


class PatientFeatureExtractor:
    """
    Convert PatientState into a deterministic numerical feature vector.

    Features include:

    - normalized current vital values
    - explicit missingness indicators
    - explicit imputation indicators
    - short-term change from the previous observation
    - static patient context when available

    Missing values are represented numerically as zero after normalization,
    while their missingness is preserved by a separate feature.
    """

    def __init__(self) -> None:
        self._feature_names = self._build_feature_names()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    def extract(self, state: PatientState) -> FeatureVector:
        """Extract the complete feature vector from a PatientState."""

        names: list[str] = []
        values: list[float] = []

        for vital_name in CORE_VITALS:
            observation = state.get_vital(vital_name)

            current_value = (
                float(observation.value)
                if observation.value is not None
                else 0.0
            )

            center, scale = _DEFAULT_SCALES[vital_name]

            if observation.value is None or observation.is_missing:
                normalized_value = 0.0
            else:
                normalized_value = (current_value - center) / scale

            names.append(f"{vital_name}.value")
            values.append(normalized_value)

            names.append(f"{vital_name}.missing")
            values.append(
                1.0
                if observation.value is None or observation.is_missing
                else 0.0
            )

            names.append(f"{vital_name}.imputed")
            values.append(
                1.0 if observation.is_imputed else 0.0
            )

            previous_value = self._previous_vital_value(
                state,
                vital_name,
            )

            if (
                previous_value is None
                or observation.value is None
                or observation.is_missing
            ):
                delta = 0.0
            else:
                delta = (current_value - previous_value) / scale

            names.append(f"{vital_name}.delta")
            values.append(delta)

        self._append_static_context(state, names, values)

        return FeatureVector(
            names=tuple(names),
            values=tuple(values),
        )

    def _append_static_context(
        self,
        state: PatientState,
        names: list[str],
        values: list[float],
    ) -> None:
        """Append available static patient context."""

        context = state.static_context

        age = getattr(context, "age", None)

        names.append("Age")
        if age is None:
            values.append(0.0)
        else:
            center, scale = _AGE_SCALE
            values.append((float(age) - center) / scale)

        gender = getattr(context, "gender", None)

        names.append("Gender")
        values.append(self._encode_gender(gender))

        unit1 = getattr(context, "unit1", None)

        names.append("Unit1")
        values.append(self._encode_binary(unit1))

        unit2 = getattr(context, "unit2", None)

        names.append("Unit2")
        values.append(self._encode_binary(unit2))

    @staticmethod
    def _previous_vital_value(
        state: PatientState,
        vital_name: str,
    ) -> float | None:
        """
        Retrieve the previous value for a vital when state history exists.

        PatientState implementations may expose history as snapshots.
        Unsupported history representations simply result in no delta.
        """

        history = getattr(state, "recent_history", None)

        if not history:
            return None

        for previous_state in reversed(history):
            if isinstance(previous_state, PatientState):
                observation = previous_state.get_vital(vital_name)

                if (
                    observation.value is not None
                    and not observation.is_missing
                ):
                    return float(observation.value)

            elif isinstance(previous_state, dict):
                value = PatientFeatureExtractor._value_from_mapping(
                    previous_state,
                    vital_name,
                )

                if value is not None:
                    return value

        return None

    @staticmethod
    def _value_from_mapping(
        snapshot: dict,
        vital_name: str,
    ) -> float | None:
        """Extract a vital from dictionary-based historical snapshots."""

        vitals = snapshot.get("current_vitals")

        if not isinstance(vitals, dict):
            return None

        observation = vitals.get(vital_name)

        if isinstance(observation, dict):
            if observation.get("is_missing", False):
                return None

            value = observation.get("value")
        else:
            value = observation

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _encode_gender(value: object) -> float:
        """Encode the dataset's binary gender field."""

        if value is None:
            return 0.0

        if isinstance(value, bool):
            return float(value)

        text = str(value).strip().lower()

        if text in {"1", "m", "male"}:
            return 1.0

        if text in {"0", "f", "female"}:
            return 0.0

        try:
            numeric = float(text)

            if numeric in (0.0, 1.0):
                return numeric
        except ValueError:
            pass

        return 0.0

    @staticmethod
    def _encode_binary(value: object) -> float:
        """Encode Unit1/Unit2-style binary context."""

        if value is None:
            return 0.0

        if isinstance(value, bool):
            return float(value)

        text = str(value).strip().lower()

        if text in {"1", "true", "yes"}:
            return 1.0

        return 0.0

    @staticmethod
    def _build_feature_names() -> tuple[str, ...]:
        """Build the stable feature ordering used by the model."""

        names: list[str] = []

        for vital_name in CORE_VITALS:
            names.extend(
                [
                    f"{vital_name}.value",
                    f"{vital_name}.missing",
                    f"{vital_name}.imputed",
                    f"{vital_name}.delta",
                ]
            )

        names.extend(
            [
                "Age",
                "Gender",
                "Unit1",
                "Unit2",
            ]
        )

        return tuple(names)