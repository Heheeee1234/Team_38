import pytest

from clinical_data_gen.preprocessing import (
    MissingValueHandler,
    ProcessedObservation,
)


def test_observed_value_is_preserved():
    handler = MissingValueHandler()

    result = handler.process("HR", 80)

    assert result == ProcessedObservation(
        value=80.0,
        is_missing=False,
        is_imputed=False,
    )


def test_short_gap_is_forward_filled():
    handler = MissingValueHandler(max_forward_fill=3)

    handler.process("HR", 80)

    result = handler.process("HR", None)

    assert result.value == 80.0
    assert result.is_missing is True
    assert result.is_imputed is True


def test_three_consecutive_missing_values_are_filled():
    handler = MissingValueHandler(max_forward_fill=3)

    handler.process("HR", 80)

    results = [
        handler.process("HR", None),
        handler.process("HR", None),
        handler.process("HR", None),
    ]

    assert [result.value for result in results] == [
        80.0,
        80.0,
        80.0,
    ]

    assert all(result.is_missing for result in results)
    assert all(result.is_imputed for result in results)


def test_fourth_missing_value_remains_missing():
    handler = MissingValueHandler(max_forward_fill=3)

    handler.process("HR", 80)

    for _ in range(3):
        handler.process("HR", None)

    result = handler.process("HR", None)

    assert result.value is None
    assert result.is_missing is True
    assert result.is_imputed is False


def test_initial_missing_value_cannot_be_imputed():
    handler = MissingValueHandler()

    result = handler.process("HR", None)

    assert result.value is None
    assert result.is_missing is True
    assert result.is_imputed is False


def test_new_observation_resets_missing_streak():
    handler = MissingValueHandler(max_forward_fill=3)

    handler.process("HR", 80)
    handler.process("HR", None)
    handler.process("HR", None)

    result = handler.process("HR", 90)

    assert result.value == 90.0
    assert result.is_missing is False
    assert result.is_imputed is False

    result = handler.process("HR", None)

    assert result.value == 90.0
    assert result.is_missing is True
    assert result.is_imputed is True


def test_vitals_are_tracked_independently():
    handler = MissingValueHandler()

    handler.process("HR", 80)
    handler.process("O2Sat", 98)

    hr_result = handler.process("HR", None)
    oxygen_result = handler.process("O2Sat", None)

    assert hr_result.value == 80.0
    assert oxygen_result.value == 98.0


def test_temperature_is_not_forward_filled():
    handler = MissingValueHandler()

    handler.process("Temp", 37.2)

    result = handler.process("Temp", None)

    assert result.value is None
    assert result.is_missing is True
    assert result.is_imputed is False


def test_temperature_remains_missing_across_long_gap():
    handler = MissingValueHandler()

    handler.process("Temp", 37.2)

    results = [
        handler.process("Temp", None)
        for _ in range(5)
    ]

    assert all(result.value is None for result in results)
    assert all(result.is_missing for result in results)
    assert all(not result.is_imputed for result in results)


def test_process_row_processes_all_core_vitals():
    handler = MissingValueHandler()

    result = handler.process_row(
        {
            "HR": 80,
            "O2Sat": 98,
            "Resp": 18,
            "SBP": 120,
            "MAP": 80,
            "DBP": 70,
            "Temp": None,
        }
    )

    assert result["HR"].value == 80.0
    assert result["O2Sat"].value == 98.0
    assert result["Resp"].value == 18.0
    assert result["SBP"].value == 120.0
    assert result["MAP"].value == 80.0
    assert result["DBP"].value == 70.0

    assert result["Temp"].value is None
    assert result["Temp"].is_missing is True
    assert result["Temp"].is_imputed is False


def test_missing_vital_not_present_in_row_is_treated_as_missing():
    handler = MissingValueHandler()

    result = handler.process_row({"HR": 80})

    assert result["HR"].value == 80.0

    for vital_name in (
        "O2Sat",
        "Resp",
        "SBP",
        "MAP",
        "DBP",
        "Temp",
    ):
        assert result[vital_name].value is None
        assert result[vital_name].is_missing is True


def test_invalid_vital_is_rejected():
    handler = MissingValueHandler()

    with pytest.raises(ValueError):
        handler.process("Lactate", 2.0)


def test_invalid_value_is_rejected():
    handler = MissingValueHandler()

    with pytest.raises(TypeError):
        handler.process("HR", "not-a-number")


def test_boolean_value_is_rejected():
    handler = MissingValueHandler()

    with pytest.raises(TypeError):
        handler.process("HR", True)


def test_negative_forward_fill_is_rejected():
    with pytest.raises(ValueError):
        MissingValueHandler(max_forward_fill=-1)


def test_zero_forward_fill_disables_imputation():
    handler = MissingValueHandler(max_forward_fill=0)

    handler.process("HR", 80)

    result = handler.process("HR", None)

    assert result.value is None
    assert result.is_missing is True
    assert result.is_imputed is False


def test_reset_clears_previous_observations():
    handler = MissingValueHandler()

    handler.process("HR", 80)

    handler.reset()

    result = handler.process("HR", None)

    assert result.value is None
    assert result.is_missing is True
    assert result.is_imputed is False


def test_numeric_strings_are_accepted():
    handler = MissingValueHandler()

    result = handler.process("HR", "80.5")

    assert result.value == 80.5
    assert result.is_missing is False
    assert result.is_imputed is False