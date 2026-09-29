"""Patient-level state management for streaming clinical telemetry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

from clinical_data_gen.preprocessing.constants import CORE_VITALS
from clinical_data_gen.preprocessing.missing_value_handler import (
    MissingValueHandler,
)
from clinical_data_gen.state.patient_state import (
    PatientState,
    StaticContext,
    VitalObservation,
)


class PatientStateManager:
    """
    Maintain isolated state for each patient in the streaming pipeline.

    Responsibilities:
    - Maintain one PatientState per patient.
    - Apply patient-specific missing-value handling to core vitals.
    - Explicitly represent missing core vitals.
    - Store laboratory observations without forward filling.
    - Maintain bounded historical state.
    - Maintain static patient context.
    - Maintain patient-specific preprocessing state.

    This class does NOT perform:
    - trend analysis
    - anomaly detection
    - risk prediction
    - alerting
    - online learning
    - clinical reasoning
    """

    def __init__(
        self,
        history_length: int = 24,
        event_history_length: int = 50,
        max_forward_fill: int = 3,
    ) -> None:
        if history_length < 0:
            raise ValueError("history_length must be >= 0")

        if event_history_length < 0:
            raise ValueError("event_history_length must be >= 0")

        if max_forward_fill < 0:
            raise ValueError("max_forward_fill must be >= 0")

        self.history_length = history_length
        self.event_history_length = event_history_length
        self.max_forward_fill = max_forward_fill

        # One state per patient.
        self.states: dict[str, PatientState] = {}

        # One missing-value handler per patient.
        #
        # This prevents forward-fill state from leaking between patients.
        self._missing_value_handlers: dict[str, MissingValueHandler] = {}

    # ------------------------------------------------------------------
    # Patient lifecycle
    # ------------------------------------------------------------------

    def get_or_create(
        self,
        patient_id: str,
        timestamp: Any = None,
    ) -> PatientState:
        """Return an existing state or create a new patient state."""

        if not patient_id:
            raise ValueError("patient_id must be non-empty")

        if patient_id not in self.states:
            self.states[patient_id] = PatientState(
                patient_id=patient_id,
                timestamp=timestamp,
                static_context=StaticContext(),
                current_vitals={},
                available_labs={},
                recent_history=[],
                baseline={},
                recent_events=[],
                metadata={},
            )

            self._missing_value_handlers[patient_id] = MissingValueHandler(
                max_forward_fill=self.max_forward_fill
            )

        return self.states[patient_id]

    def get(self, patient_id: str) -> Optional[PatientState]:
        """Return patient state if present, otherwise None."""

        return self.states.get(patient_id)

    def has_patient(self, patient_id: str) -> bool:
        """Return whether a patient is currently tracked."""

        return patient_id in self.states

    def remove_patient(self, patient_id: str) -> Optional[PatientState]:
        """
        Remove and return all state associated with a patient.

        The missing-value handler is removed at the same time so no
        preprocessing state survives after patient removal.
        """

        removed_state = self.states.pop(patient_id, None)
        self._missing_value_handlers.pop(patient_id, None)

        return removed_state

    def clear(self) -> None:
        """Remove all patients and all patient-specific preprocessing state."""

        self.states.clear()
        self._missing_value_handlers.clear()

    @property
    def patient_count(self) -> int:
        """Return the number of currently tracked patients."""

        return len(self.states)

    # ------------------------------------------------------------------
    # Main update API
    # ------------------------------------------------------------------

    def update(
        self,
        patient_id: str,
        timestamp: Any,
        vitals: Optional[Mapping[str, Optional[float]]] = None,
        labs: Optional[Mapping[str, Optional[float]]] = None,
        static_context: Optional[Mapping[str, Any] | StaticContext] = None,
        events: Optional[list[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PatientState:
        """
        Update the state of one patient.

        Parameters
        ----------
        patient_id:
            Stable patient identifier.

        timestamp:
            Timestamp associated with the current observation.

        vitals:
            Core vital observations.

        labs:
            Laboratory observations. Labs are never forward filled.

        static_context:
            Patient static context. Accepts either a StaticContext instance
            or a mapping.

        events:
            Optional events to append to the patient's event history.

        metadata:
            Optional metadata to merge into patient metadata.

        Returns
        -------
        PatientState
            The updated patient state.
        """

        is_new_patient = patient_id not in self.states

        state = self.get_or_create(
            patient_id=patient_id,
            timestamp=timestamp,
        )

        # Only snapshot an EXISTING state that actually contains data.
        #
        # A newly created PatientState has no previous observation and
        # therefore must not generate an empty history entry.
        if (
            not is_new_patient
            and self.history_length > 0
            and self._has_observation_data(state)
        ):
            self._snapshot_current_state(state)

        state.timestamp = timestamp

        if static_context is not None:
            self._update_static_context(
                state=state,
                static_context=static_context,
            )

        if vitals is not None:
            self._update_vitals(
                patient_id=patient_id,
                state=state,
                vitals=vitals,
            )

        if labs is not None:
            self._update_labs(
                state=state,
                labs=labs,
            )

        if events:
            for event in events:
                state.add_event(
                    deepcopy(dict(event)),
                    max_length=self.event_history_length,
                )

        if metadata:
            state.metadata.update(dict(metadata))

        return state

    # ------------------------------------------------------------------
    # Static context
    # ------------------------------------------------------------------

    def _update_static_context(
        self,
        state: PatientState,
        static_context: Mapping[str, Any] | StaticContext,
    ) -> None:
        """
        Update static patient context.

        Accepts either:
        - StaticContext
        - mapping with the StaticContext field names

        Missing fields in a mapping preserve the existing values.
        """

        if isinstance(static_context, StaticContext):
            state.static_context = static_context
            return

        current = state.static_context

        state.static_context = StaticContext(
            age=(
                static_context["age"]
                if "age" in static_context
                else current.age
            ),
            gender=(
                static_context["gender"]
                if "gender" in static_context
                else current.gender
            ),
            unit1=(
                static_context["unit1"]
                if "unit1" in static_context
                else current.unit1
            ),
            unit2=(
                static_context["unit2"]
                if "unit2" in static_context
                else current.unit2
            ),
            hosp_adm_time=(
                static_context["hosp_adm_time"]
                if "hosp_adm_time" in static_context
                else current.hosp_adm_time
            ),
        )

    # ------------------------------------------------------------------
    # Vitals
    # ------------------------------------------------------------------

    def _update_vitals(
        self,
        patient_id: str,
        state: PatientState,
        vitals: Mapping[str, Optional[float]],
    ) -> None:
        """
        Process and store core-vital observations.

        Missing-value handling is patient-specific. Explicitly absent
        core vitals are represented as missing observations.
        """

        handler = self._missing_value_handlers[patient_id]
        processed = handler.process_row(dict(vitals))

        for vital_name in CORE_VITALS:
            # A completely absent vital is genuinely missing.
            if vital_name not in vitals:
                state.set_vital(
                    vital_name,
                    VitalObservation(
                        value=None,
                        is_missing=True,
                        is_imputed=False,
                    ),
                )
                continue

            observation = processed[vital_name]

            state.set_vital(
                vital_name,
                VitalObservation(
                    value=observation.value,
                    is_missing=observation.is_missing,
                    is_imputed=observation.is_imputed,
                ),
            )

    # ------------------------------------------------------------------
    # Laboratory observations
    # ------------------------------------------------------------------

    def _update_labs(
        self,
        state: PatientState,
        labs: Mapping[str, Optional[float]],
    ) -> None:
        """
        Store laboratory observations without forward filling.

        Labs are represented using VitalObservation so that missingness is
        explicitly preserved.

        A missing laboratory result is:
            value=None
            is_missing=True
            is_imputed=False
        """

        for lab_name, value in labs.items():
            if value is None:
                state.available_labs[lab_name] = VitalObservation(
                    value=None,
                    is_missing=True,
                    is_imputed=False,
                )
                continue

            if isinstance(value, bool):
                raise TypeError(
                    f"Lab observation must be numeric or None: "
                    f"{lab_name}={value!r}"
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Lab observation must be numeric or None: "
                    f"{lab_name}={value!r}"
                ) from exc

            state.available_labs[lab_name] = VitalObservation(
                value=numeric_value,
                is_missing=False,
                is_imputed=False,
            )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _has_observation_data(
        self,
        state: PatientState,
    ) -> bool:
        """
        Return whether the state contains data worth snapshotting.
        """

        return bool(
            state.current_vitals
            or state.available_labs
            or state.recent_events
            or state.baseline
        )

    def _snapshot_current_state(
        self,
        state: PatientState,
    ) -> None:
        """
        Store the current state as a bounded historical snapshot.

        Deep copies are used so later updates cannot mutate historical
        records.
        """

        snapshot = {
            "timestamp": state.timestamp,
            "current_vitals": deepcopy(state.current_vitals),
            "available_labs": deepcopy(state.available_labs),
            "baseline": deepcopy(state.baseline),
        }

        state.add_history(
            snapshot,
            max_length=self.history_length,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        patient_id: str,
        baseline: Mapping[str, Any],
    ) -> PatientState:
        """Set patient baseline information."""

        state = self.get_or_create(patient_id)

        state.set_baseline(
            deepcopy(dict(baseline))
        )

        return state

    def add_event(
        self,
        patient_id: str,
        event: Mapping[str, Any],
    ) -> PatientState:
        """Add an event to the patient's bounded event history."""

        state = self.get_or_create(patient_id)

        state.add_event(
            deepcopy(dict(event)),
            max_length=self.event_history_length,
        )

        return state