from clinical_data_gen.state import (
    PatientState,
    StaticContext,
)


def test_patient_state_creation():

    state = PatientState(
        patient_id="patient_001",
        timestamp=10.0,
        static_context=StaticContext(
            age=65.0,
            gender=1,
            unit1=1,
            unit2=0,
            hosp_adm_time=-5.0,
        ),
    )

    assert state.patient_id == "patient_001"
    assert state.timestamp == 10.0
    assert state.static_context.age == 65.0


def test_set_vital():

    state = PatientState(
        patient_id="patient_001"
    )

    state.set_vital(
        "HR",
        82.0,
        is_missing=False,
        is_imputed=False,
    )

    assert state.get_vital("HR") == 82.0
    assert state.is_vital_missing("HR") is False
    assert state.is_vital_imputed("HR") is False


def test_imputed_vital():

    state = PatientState(
        patient_id="patient_001"
    )

    state.set_vital(
        "HR",
        82.0,
        is_missing=True,
        is_imputed=True,
    )

    assert state.get_vital("HR") == 82.0
    assert state.is_vital_missing("HR") is True
    assert state.is_vital_imputed("HR") is True


def test_missing_vital():

    state = PatientState(
        patient_id="patient_001"
    )

    state.set_vital(
        "HR",
        None,
        is_missing=True,
        is_imputed=False,
    )

    assert state.get_vital("HR") is None
    assert state.is_vital_missing("HR") is True


def test_lab():

    state = PatientState(
        patient_id="patient_001"
    )

    state.set_lab(
        "Lactate",
        2.1,
    )

    assert state.available_labs["Lactate"].value == 2.1


def test_history_limit():

    state = PatientState(
        patient_id="patient_001"
    )

    for i in range(30):
        state.add_history(
            {"timestamp": i},
            max_length=24,
        )

    assert len(state.recent_history) == 24
    assert state.recent_history[0]["timestamp"] == 6


def test_event_storage():

    state = PatientState(
        patient_id="patient_001"
    )

    state.add_event({
        "type": "alert",
        "severity": "high",
    })

    assert len(state.recent_events) == 1
    assert state.recent_events[0]["type"] == "alert"


def test_baseline():

    state = PatientState(
        patient_id="patient_001"
    )

    state.set_baseline(
        "HR",
        {
            "mean": 75.0,
            "std": 8.0,
        },
    )

    assert state.baseline["HR"]["mean"] == 75.0


def test_serialization():

    state = PatientState(
        patient_id="patient_001",
        timestamp=10.0,
    )

    state.set_vital(
        "HR",
        82.0,
    )

    result = state.to_dict()

    assert result["patient_id"] == "patient_001"
    assert result["current_vitals"]["HR"]["value"] == 82.0