from clinical_data_gen.ingestion import (
    PSV_COLUMNS,
    PSVIngestor,
)
from clinical_data_gen.state import PatientStateManager


def make_row(**overrides):
    """Create a complete valid PSV row for testing."""

    row = {
        column: None
        for column in PSV_COLUMNS
    }

    row.update(
        {
            "HR": "80",
            "O2Sat": "98",
            "Temp": "37.0",
            "SBP": "120",
            "MAP": "80",
            "DBP": "70",
            "Resp": "18",
            "Age": "65",
            "Gender": "1",
            "Unit1": "1",
            "Unit2": "0",
            "HospAdmTime": "-5.5",
            "ICULOS": "1",
            "SepsisLabel": "0",
        }
    )

    row.update(overrides)

    return row


def test_parse_complete_psv_row():
    ingestor = PSVIngestor(PatientStateManager())

    parsed = ingestor.parse_row(make_row())

    assert parsed.timestamp == 1.0
    assert parsed.vitals["HR"] == 80.0
    assert parsed.vitals["O2Sat"] == 98.0
    assert parsed.vitals["Temp"] == 37.0

    assert parsed.static_context.age == 65.0
    assert parsed.static_context.gender == 1
    assert parsed.static_context.unit1 == 1
    assert parsed.static_context.unit2 == 0
    assert parsed.static_context.hosp_adm_time == -5.5

    assert parsed.sepsis_label == 0


def test_missing_psv_values_become_none():
    ingestor = PSVIngestor(PatientStateManager())

    parsed = ingestor.parse_row(
        make_row(
            HR="",
            O2Sat="NaN",
            Temp=None,
            Lactate="",
        )
    )

    assert parsed.vitals["HR"] is None
    assert parsed.vitals["O2Sat"] is None
    assert parsed.vitals["Temp"] is None
    assert parsed.labs["Lactate"] is None


def test_ingest_row_updates_patient_state():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    state, label = ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            HR="85",
            O2Sat="97",
            ICULOS="4",
        ),
    )

    assert state.patient_id == "patient-1"
    assert state.timestamp == 4.0
    assert state.get_vital("HR").value == 85.0
    assert state.get_vital("O2Sat").value == 97.0
    assert label == 0


def test_sepsis_label_is_not_stored_in_patient_state():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    state, label = ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(SepsisLabel="1"),
    )

    assert label == 1
    assert not hasattr(state, "sepsis_label")
    assert "SepsisLabel" not in state.metadata


def test_short_gap_is_processed_through_manager():
    manager = PatientStateManager(max_forward_fill=3)
    ingestor = PSVIngestor(manager)

    ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            HR="80",
            ICULOS="1",
        ),
    )

    state, _ = ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            HR="",
            ICULOS="2",
        ),
    )

    observation = state.get_vital("HR")

    assert observation.value == 80.0
    assert observation.is_missing is True
    assert observation.is_imputed is True


def test_temperature_missingness_is_preserved():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            Temp="37.2",
            ICULOS="1",
        ),
    )

    state, _ = ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            Temp="",
            ICULOS="2",
        ),
    )

    observation = state.get_vital("Temp")

    assert observation.value is None
    assert observation.is_missing is True
    assert observation.is_imputed is False


def test_labs_are_ingested_without_forward_fill():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            Lactate="2.5",
            ICULOS="1",
        ),
    )

    state, _ = ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            Lactate="",
            ICULOS="2",
        ),
    )

    observation = state.available_labs["Lactate"]

    assert observation.value is None
    assert observation.is_missing is True
    assert observation.is_imputed is False


def test_static_context_is_ingested():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    state, _ = ingestor.ingest_row(
        patient_id="patient-1",
        row=make_row(
            Age="72",
            Gender="0",
            Unit1="0",
            Unit2="1",
            HospAdmTime="-12.0",
        ),
    )

    assert state.static_context.age == 72.0
    assert state.static_context.gender == 0
    assert state.static_context.unit1 == 0
    assert state.static_context.unit2 == 1
    assert state.static_context.hosp_adm_time == -12.0


def test_multiple_rows_are_ingested_in_order():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    rows = [
        make_row(HR="80", ICULOS="1", SepsisLabel="0"),
        make_row(HR="82", ICULOS="2", SepsisLabel="0"),
        make_row(HR="90", ICULOS="3", SepsisLabel="1"),
    ]

    results = ingestor.ingest_rows(
        patient_id="patient-1",
        rows=rows,
    )

    assert len(results) == 3

    state, label = results[-1]

    assert state.timestamp == 3.0
    assert state.get_vital("HR").value == 90.0
    assert label == 1
    assert len(state.recent_history) == 2


def test_missing_required_columns_are_rejected():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    row = make_row()
    del row["ICULOS"]

    try:
        ingestor.parse_row(row)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "ICULOS" in str(exc)


def test_invalid_sepsis_label_is_rejected():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    try:
        ingestor.parse_row(make_row(SepsisLabel="2"))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "SepsisLabel" in str(exc)


def test_invalid_timestamp_is_rejected():
    manager = PatientStateManager()
    ingestor = PSVIngestor(manager)

    try:
        ingestor.parse_row(make_row(ICULOS="invalid"))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "ICULOS" in str(exc)