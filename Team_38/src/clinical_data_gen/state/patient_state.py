from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from clinical_data_gen.preprocessing.constants import CORE_VITALS



# ============================================================
# Supporting structures
# ============================================================

@dataclass
class VitalObservation:
    """
    A single vital-sign observation.

    value:
        Observed or imputed value.

    is_missing:
        True if the original observation was missing.

    is_imputed:
        True if the value was filled by MissingValueHandler.
    """

    value: Optional[float] = None
    is_missing: bool = False
    is_imputed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "is_missing": self.is_missing,
            "is_imputed": self.is_imputed,
        }


@dataclass
class StaticContext:
    """
    Patient/context information that does not normally change
    at every timestep.
    """

    age: Optional[float] = None
    gender: Optional[int] = None
    unit1: Optional[int] = None
    unit2: Optional[int] = None
    hosp_adm_time: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "gender": self.gender,
            "unit1": self.unit1,
            "unit2": self.unit2,
            "hosp_adm_time": self.hosp_adm_time,
        }


@dataclass
class PatientState:
    """
    Current state of one patient at a particular timestep.

    PatientState is a state container.

    It does NOT:
        - calculate anomaly scores
        - calculate risk
        - make clinical decisions
        - generate alerts
        - run ML models

    Those responsibilities belong to downstream components.
    """

    patient_id: str

    timestamp: Optional[float] = None

    # --------------------------------------------------------
    # Patient/context information
    # --------------------------------------------------------

    static_context: StaticContext = field(
        default_factory=StaticContext
    )

    # --------------------------------------------------------
    # Current physiological observations
    # --------------------------------------------------------

    current_vitals: dict[str, VitalObservation] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Labs that happen to be available at this timestep
    # --------------------------------------------------------

    available_labs: dict[str, VitalObservation] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Recent temporal history
    #
    # Each entry represents a previous PatientState snapshot
    # or a lightweight observation dictionary.
    # --------------------------------------------------------

    recent_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Patient-specific baseline
    #
    # Populated later by baseline estimation.
    # --------------------------------------------------------

    baseline: dict[str, Any] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Recent events / alerts / feedback
    # --------------------------------------------------------

    recent_events: list[dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Additional metadata
    # --------------------------------------------------------

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================
    # Construction helpers
    # ========================================================

    def set_vital(
        self,
        name: str,
        value: Optional[float],
        *,
        is_missing: bool = False,
        is_imputed: bool = False,
    ) -> None:
        """
        Add or update a current vital observation.
        """

        if name not in CORE_VITALS:
            raise ValueError(
                f"Unknown core vital: {name}"
            )

        self.current_vitals[name] = VitalObservation(
            value=value,
            is_missing=is_missing,
            is_imputed=is_imputed,
        )

    def set_lab(
        self,
        name: str,
        value: Optional[float],
        *,
        is_missing: bool = False,
        is_imputed: bool = False,
    ) -> None:
        """
        Add or update an available laboratory observation.

        Labs are intentionally handled separately from core
        streaming vitals.
        """

        self.available_labs[name] = VitalObservation(
            value=value,
            is_missing=is_missing,
            is_imputed=is_imputed,
        )

    # ========================================================
    # History
    # ========================================================

    def add_history(
        self,
        observation: dict[str, Any],
        *,
        max_length: int = 24,
    ) -> None:
        """
        Add a previous observation to recent history.

        max_length:
            Maximum number of historical observations retained.
        """

        self.recent_history.append(observation)

        if len(self.recent_history) > max_length:
            self.recent_history = self.recent_history[-max_length:]

    # ========================================================
    # Events
    # ========================================================

    def add_event(
        self,
        event: dict[str, Any],
        *,
        max_length: int = 50,
    ) -> None:
        """
        Store a recent event such as an alert, clinician decision,
        investigation result, or feedback event.
        """

        self.recent_events.append(event)

        if len(self.recent_events) > max_length:
            self.recent_events = self.recent_events[-max_length:]

    # ========================================================
    # Baseline
    # ========================================================

    def set_baseline(
        self,
        name: str,
        baseline_value: Any,
    ) -> None:
        """
        Store a patient-specific baseline value.

        Baseline calculation is handled elsewhere.
        """

        self.baseline[name] = baseline_value

    # ========================================================
    # Access helpers
    # ========================================================

    def get_vital(
        self,
        name: str,
    ) -> Optional[float]:
        """
        Return the current value of a vital.

        Returns None when the vital is unavailable.
        """

        observation = self.current_vitals.get(name)

        if observation is None:
            return None

        return observation.value

    def is_vital_missing(self, name: str) -> bool:
        """Return whether the current vital observation is missing."""
        observation = self.current_vitals.get(name)

        if observation is None:
            return True

        return observation.value is None or observation.is_missing

    def is_vital_imputed(self, name: str) -> bool:
        """Return whether the current vital value was forward-filled."""
        observation = self.current_vitals.get(name)

        if observation is None:
            return False

        return observation.is_imputed

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert PatientState into a JSON-serializable dictionary.
        """

        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,

            "static_context": (
                self.static_context.to_dict()
            ),

            "current_vitals": {
                name: observation.to_dict()
                for name, observation
                in self.current_vitals.items()
            },

            "available_labs": {
                name: observation.to_dict()
                for name, observation
                in self.available_labs.items()
            },

            "recent_history": self.recent_history,

            "baseline": self.baseline,

            "recent_events": self.recent_events,

            "metadata": self.metadata,
        }