from clinical_data_gen.state import (
    PatientStateManager,
    StaticContext,
)


def test_manager_creates_patient_state():
    manager = PatientStateManager()

    state = manager.get_or_create(
        patient_id="patient-1",
        timestamp="2026-01-01T00:00:00",
    )

    assert state.patient_id == "patient-1"
    assert state.timestamp == "2026-01-01T00:00:00"
    assert manager.patient_count == 1


def test_manager_returns_existing_state():
    manager = PatientStateManager()

    first = manager.get_or_create("patient-1")
    second = manager.get_or_create("patient-1")

    assert first is second
    assert manager.patient_count == 1


def test_manager_updates_vitals():
    manager = PatientStateManager()

    state = manager.update(
        patient_id="patient-1",
        timestamp="2026-01-01T01:00:00",
        vitals={
            "HR": 80,
            "O2Sat": 98,
            "Resp": 18,
        },
    )

    assert state.get_vital("HR").value == 80.0
    assert state.get_vital("O2Sat").value == 98.0
    assert state.get_vital("Resp").value == 18.0


def test_short_gap_is_imputed_by_manager():
    manager = PatientStateManager(max_forward_fill=3)

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"HR": 80},
    )

    state = manager.update(
        patient_id="patient-1",
        timestamp="t2",
        vitals={"HR": None},
    )

    observation = state.get_vital("HR")

    assert observation.value == 80.0
    assert observation.is_missing is True
    assert observation.is_imputed is True


def test_long_gap_remains_missing():
    manager = PatientStateManager(max_forward_fill=3)

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"HR": 80},
    )

    for index in range(1, 5):
        state = manager.update(
            patient_id="patient-1",
            timestamp=f"t{index + 1}",
            vitals={"HR": None},
        )

    observation = state.get_vital("HR")

    assert observation.value is None
    assert observation.is_missing is True
    assert observation.is_imputed is False


def test_temperature_is_not_forward_filled():
    manager = PatientStateManager()

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"Temp": 37.2},
    )

    state = manager.update(
        patient_id="patient-1",
        timestamp="t2",
        vitals={"Temp": None},
    )

    observation = state.get_vital("Temp")

    assert observation.value is None
    assert observation.is_missing is True
    assert observation.is_imputed is False

def test_patient_histories_are_independent():
    manager = PatientStateManager()

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"HR": 80},
    )

    manager.update(
        patient_id="patient-1",
        timestamp="t2",
        vitals={"HR": 81},
    )

    manager.update(
        patient_id="patient-2",
        timestamp="t1",
        vitals={"HR": 100},
    )

    patient_1 = manager.get("patient-1")
    patient_2 = manager.get("patient-2")

    assert patient_1 is not None
    assert patient_2 is not None

    assert patient_1.get_vital("HR").value == 81.0
    assert patient_2.get_vital("HR").value == 100.0

    assert len(patient_1.recent_history) == 1
    assert len(patient_2.recent_history) == 0


def test_missing_value_state_is_patient_specific():
    manager = PatientStateManager(max_forward_fill=3)

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"HR": 80},
    )

    patient_1 = manager.update(
        patient_id="patient-1",
        timestamp="t2",
        vitals={"HR": None},
    )

    patient_2 = manager.update(
        patient_id="patient-2",
        timestamp="t1",
        vitals={"HR": None},
    )

    assert patient_1.get_vital("HR").value == 80.0
    assert patient_1.get_vital("HR").is_imputed is True

    assert patient_2.get_vital("HR").value is None
    assert patient_2.get_vital("HR").is_imputed is False


def test_new_observation_resets_imputation_streak():
    manager = PatientStateManager(max_forward_fill=3)

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"HR": 80},
    )

    manager.update(
        patient_id="patient-1",
        timestamp="t2",
        vitals={"HR": None},
    )

    manager.update(
        patient_id="patient-1",
        timestamp="t3",
        vitals={"HR": 90},
    )

    state = manager.update(
        patient_id="patient-1",
        timestamp="t4",
        vitals={"HR": None},
    )

    observation = state.get_vital("HR")

    assert observation.value == 90.0
    assert observation.is_missing is True
    assert observation.is_imputed is True


def test_labs_are_not_forward_filled():
    manager = PatientStateManager()

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        labs={"Lactate": 2.0},
    )

    state = manager.update(
        patient_id="patient-1",
        timestamp="t2",
        labs={"Lactate": None},
    )

    observation = state.available_labs["Lactate"]

    assert observation.value is None
    assert observation.is_missing is True
    assert observation.is_imputed is False


def test_labs_are_stored_when_observed():
    manager = PatientStateManager()

    state = manager.update(
        patient_id="patient-1",
        timestamp="t1",
        labs={
            "Lactate": 2.1,
            "Creatinine": 1.2,
        },
    )

    assert state.available_labs["Lactate"].value == 2.1
    assert state.available_labs["Creatinine"].value == 1.2


def test_static_context_is_updated():
    manager = PatientStateManager()

    context = StaticContext(
        age=65,
        gender=1,
        unit1=1,
        unit2=0,
        hosp_adm_time=-5.5,
    )

    state = manager.update(
        patient_id="patient-1",
        timestamp="t1",
        static_context=context,
    )

    assert state.static_context.age == 65
    assert state.static_context.gender == 1
    assert state.static_context.unit1 == 1
    assert state.static_context.unit2 == 0


def test_events_are_stored():
    manager = PatientStateManager()

    state = manager.update(
        patient_id="patient-1",
        timestamp="t1",
        events=[
            {
                "type": "observation",
                "description": "Heart rate increased",
            }
        ],
    )

    assert len(state.recent_events) == 1
    assert state.recent_events[0]["type"] == "observation"


def test_remove_patient_clears_state_and_preprocessor():
    manager = PatientStateManager()

    manager.update(
        patient_id="patient-1",
        timestamp="t1",
        vitals={"HR": 80},
    )

    removed = manager.remove_patient("patient-1")

    assert removed is not None
    assert manager.has_patient("patient-1") is False
    assert manager.patient_count == 0


def test_clear_removes_all_patients():
    manager = PatientStateManager()

    manager.update("patient-1", "t1", vitals={"HR": 80})
    manager.update("patient-2", "t1", vitals={"HR": 90})

    manager.clear()

    assert manager.patient_count == 0
    assert manager.has_patient("patient-1") is False
    assert manager.has_patient("patient-2") is False