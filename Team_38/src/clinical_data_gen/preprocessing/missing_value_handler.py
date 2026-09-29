from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from clinical_data_gen.preprocessing.constants import CORE_VITALS


# Core streaming vitals for which short-gap forward filling is acceptable.
FORWARD_FILL_VITALS = frozenset(
    {
        "HR",
        "O2Sat",
        "Resp",
        "SBP",
        "MAP",
        "DBP",
    }
)


@dataclass(frozen=True)
class ProcessedObservation:
    """
    Result of processing a single physiological observation.

    Attributes:
        value:
            Value available to downstream components. This may be an
            imputed value when a short missing gap is filled.
        is_missing:
            True when the original observation was missing.
        is_imputed:
            True when `value` was generated through imputation.
    """

    value: Optional[float]
    is_missing: bool
    is_imputed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the observation in serializable dictionary form."""
        return {
            "value": self.value,
            "is_missing": self.is_missing,
            "is_imputed": self.is_imputed,
        }


class MissingValueHandler:
    """
    Handle missing physiological observations in streaming data.

    The handler applies conservative, clinically-oriented preprocessing:

    * HR, O2Sat, Resp, SBP, MAP and DBP:
      short gaps are forward-filled up to `max_forward_fill` timesteps.

    * Temp:
      missing observations are preserved. Temperature is not forward-filled.

    * Long gaps:
      remain missing.

    * Initial missing observations:
      remain missing because no previous observation exists.

    Original missingness is always preserved through `is_missing`.
    """

    DEFAULT_MAX_FORWARD_FILL = 3

    def __init__(
        self,
        max_forward_fill: int = DEFAULT_MAX_FORWARD_FILL,
    ) -> None:
        if max_forward_fill < 0:
            raise ValueError("max_forward_fill must be non-negative")

        self.max_forward_fill = max_forward_fill

        # Last genuinely observed value for each forward-fill-enabled vital.
        self._last_observed: dict[str, float] = {}

        # Number of consecutive missing observations after the last
        # genuinely observed value.
        self._missing_streak: dict[str, int] = {}

    def reset(self) -> None:
        """
        Reset all patient/timestep-specific preprocessing state.

        A handler instance should be reset when starting a new patient
        stream if the same handler instance is reused.
        """
        self._last_observed.clear()
        self._missing_streak.clear()

    def process(
        self,
        vital_name: str,
        value: Optional[float],
    ) -> ProcessedObservation:
        """
        Process one core-vital observation.

        Args:
            vital_name:
                Name of the core vital.
            value:
                Raw value. `None` represents a missing observation.

        Returns:
            ProcessedObservation containing the processed value and
            missingness metadata.

        Raises:
            ValueError:
                If `vital_name` is not a supported core vital.
            TypeError:
                If a non-numeric value is supplied.
        """
        self._validate_vital(vital_name)

        if value is not None:
            numeric_value = self._to_float(value)

            self._record_observation(vital_name, numeric_value)

            return ProcessedObservation(
                value=numeric_value,
                is_missing=False,
                is_imputed=False,
            )

        return self._process_missing(vital_name)

    def process_row(
        self,
        observations: dict[str, Optional[float]],
    ) -> dict[str, ProcessedObservation]:
        """
        Process all core vitals for one timestep.

        Any core vital absent from `observations` is treated as missing.

        Args:
            observations:
                Mapping from vital name to raw value.

        Returns:
            Processed observations for every core vital.
        """
        return {
            vital_name: self.process(
                vital_name,
                observations.get(vital_name),
            )
            for vital_name in CORE_VITALS
        }

    def _process_missing(
        self,
        vital_name: str,
    ) -> ProcessedObservation:
        """Process a missing observation according to the vital policy."""

        # Temperature is deliberately not forward-filled.
        if vital_name not in FORWARD_FILL_VITALS:
            return ProcessedObservation(
                value=None,
                is_missing=True,
                is_imputed=False,
            )

        streak = self._missing_streak.get(vital_name, 0) + 1
        self._missing_streak[vital_name] = streak

        last_value = self._last_observed.get(vital_name)

        if (
            last_value is not None
            and streak <= self.max_forward_fill
        ):
            return ProcessedObservation(
                value=last_value,
                is_missing=True,
                is_imputed=True,
            )

        return ProcessedObservation(
            value=None,
            is_missing=True,
            is_imputed=False,
        )

    def _record_observation(
        self,
        vital_name: str,
        value: float,
    ) -> None:
        """Record a genuinely observed value and reset its gap counter."""

        if vital_name in FORWARD_FILL_VITALS:
            self._last_observed[vital_name] = value
            self._missing_streak[vital_name] = 0

    @staticmethod
    def _validate_vital(vital_name: str) -> None:
        """Validate that the supplied name is a supported core vital."""

        if vital_name not in CORE_VITALS:
            raise ValueError(
                f"Unsupported core vital: {vital_name!r}. "
                f"Expected one of: {', '.join(CORE_VITALS)}"
            )

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert an observed value to float with clear validation."""

        if isinstance(value, bool):
            raise TypeError("Boolean values are not valid vital observations")

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Vital observation must be numeric or None, got {value!r}"
            ) from exc