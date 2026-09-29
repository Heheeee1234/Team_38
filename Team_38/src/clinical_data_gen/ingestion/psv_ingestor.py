from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from clinical_data_gen.state import PatientState, PatientStateManager, StaticContext
from clinical_data_gen.state.patient_state import CORE_VITALS


# ---------------------------------------------------------------------------
# PSV schema
# ---------------------------------------------------------------------------

PSV_COLUMNS: tuple[str, ...] = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel",
)

LAB_COLUMNS: frozenset[str] = frozenset(
    {
        "EtCO2",
        "BaseExcess",
        "HCO3",
        "FiO2",
        "pH",
        "PaCO2",
        "SaO2",
        "AST",
        "BUN",
        "Alkalinephos",
        "Calcium",
        "Chloride",
        "Creatinine",
        "Bilirubin_direct",
        "Glucose",
        "Lactate",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "Bilirubin_total",
        "TroponinI",
        "Hct",
        "Hgb",
        "PTT",
        "WBC",
        "Fibrinogen",
        "Platelets",
    }
)

STATIC_COLUMNS: frozenset[str] = frozenset(
    {
        "Age",
        "Gender",
        "Unit1",
        "Unit2",
        "HospAdmTime",
    }
)

GROUND_TRUTH_COLUMNS: frozenset[str] = frozenset(
    {
        "SepsisLabel",
    }
)

TIMESTAMP_COLUMN = "ICULOS"


@dataclass(frozen=True)
class ParsedPSVRow:
    """
    Parsed representation of one PSV timestep.

    `sepsis_label` is retained strictly as evaluation ground truth and
    is never passed into PatientState.
    """

    timestamp: float
    vitals: dict[str, Optional[float]]
    labs: dict[str, Optional[float]]
    static_context: StaticContext
    sepsis_label: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the parsed row."""

        return {
            "timestamp": self.timestamp,
            "vitals": self.vitals,
            "labs": self.labs,
            "static_context": self.static_context.to_dict(),
            "sepsis_label": self.sepsis_label,
        }


class PSVIngestor:
    """
    Convert PhysioNet PSV rows into PatientStateManager updates.

    Responsibilities:
        - Validate the PSV schema.
        - Parse raw string values.
        - Separate vitals, labs, static context and ground truth.
        - Send observations to PatientStateManager.
        - Keep SepsisLabel outside PatientState.

    This class does not perform:
        - anomaly detection
        - trend analysis
        - risk scoring
        - alert generation
        - model inference
    """

    def __init__(
        self,
        state_manager: PatientStateManager,
    ) -> None:
        self.state_manager = state_manager

    def ingest_row(
        self,
        patient_id: str,
        row: Mapping[str, Any],
    ) -> tuple[PatientState, int]:
        """
        Parse and ingest one PSV timestep.

        Args:
            patient_id:
                Identifier of the patient whose timestep is being processed.

            row:
                Mapping containing one complete PSV row.

        Returns:
            Tuple containing:
                - updated PatientState
                - SepsisLabel ground truth for this timestep

        Raises:
            ValueError:
                If required columns are missing or values are invalid.
        """

        parsed = self.parse_row(row)

        state = self.state_manager.update(
            patient_id=patient_id,
            timestamp=parsed.timestamp,
            vitals=parsed.vitals,
            labs=parsed.labs,
            static_context=parsed.static_context,
        )

        return state, parsed.sepsis_label

    def parse_row(
        self,
        row: Mapping[str, Any],
    ) -> ParsedPSVRow:
        """
        Parse and validate one PSV row without updating patient state.
        """

        self._validate_schema(row)

        return ParsedPSVRow(
            timestamp=self._parse_required_float(
                row[TIMESTAMP_COLUMN],
                TIMESTAMP_COLUMN,
            ),
            vitals=self._parse_observations(
                row,
                CORE_VITALS,
            ),
            labs=self._parse_observations(
                row,
                LAB_COLUMNS,
            ),
            static_context=self._parse_static_context(row),
            sepsis_label=self._parse_sepsis_label(row["SepsisLabel"]),
        )

    def ingest_rows(
        self,
        patient_id: str,
        rows: list[Mapping[str, Any]],
    ) -> list[tuple[PatientState, int]]:
        """
        Ingest an ordered sequence of PSV rows for one patient.

        Rows are expected to be supplied in chronological ICULOS order.
        """

        results: list[tuple[PatientState, int]] = []

        for row in rows:
            results.append(
                self.ingest_row(
                    patient_id=patient_id,
                    row=row,
                )
            )

        return results

    @staticmethod
    def _validate_schema(
        row: Mapping[str, Any],
    ) -> None:
        """Ensure every required PSV column is present."""

        missing_columns = [
            column
            for column in PSV_COLUMNS
            if column not in row
        ]

        if missing_columns:
            raise ValueError(
                "PSV row is missing required columns: "
                + ", ".join(missing_columns)
            )

    @staticmethod
    def _parse_observations(
        row: Mapping[str, Any],
        columns: Any,
    ) -> dict[str, Optional[float]]:
        """Parse a collection of numeric observations."""

        return {
            column: PSVIngestor._parse_optional_float(
                row[column],
                column,
            )
            for column in columns
        }

    @staticmethod
    def _parse_static_context(
        row: Mapping[str, Any],
    ) -> StaticContext:
        """Parse patient-level static/context fields."""

        return StaticContext(
            age=PSVIngestor._parse_optional_float(
                row["Age"],
                "Age",
            ),
            gender=PSVIngestor._parse_optional_int(
                row["Gender"],
                "Gender",
            ),
            unit1=PSVIngestor._parse_optional_int(
                row["Unit1"],
                "Unit1",
            ),
            unit2=PSVIngestor._parse_optional_int(
                row["Unit2"],
                "Unit2",
            ),
            hosp_adm_time=PSVIngestor._parse_optional_float(
                row["HospAdmTime"],
                "HospAdmTime",
            ),
        )

    @staticmethod
    def _parse_sepsis_label(value: Any) -> int:
        """Parse the binary SepsisLabel ground truth."""

        if value is None or str(value).strip() == "":
            raise ValueError("SepsisLabel cannot be missing")

        try:
            label = int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid SepsisLabel: {value!r}"
            ) from exc

        if label not in (0, 1):
            raise ValueError(
                f"SepsisLabel must be 0 or 1, got {label}"
            )

        return label

    @staticmethod
    def _parse_required_float(
        value: Any,
        column: str,
    ) -> float:
        """Parse a required numeric field."""

        if value is None or str(value).strip() == "":
            raise ValueError(
                f"Required column {column!r} cannot be missing"
            )

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value for {column!r}: {value!r}"
            ) from exc

    @staticmethod
    def _parse_optional_float(
        value: Any,
        column: str,
    ) -> Optional[float]:
        """Parse an optional numeric field."""

        if value is None:
            return None

        text = str(value).strip()

        if text == "" or text.lower() in {
            "nan",
            "na",
            "null",
            "none",
        }:
            return None

        try:
            return float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value for {column!r}: {value!r}"
            ) from exc

    @staticmethod
    def _parse_optional_int(
        value: Any,
        column: str,
    ) -> Optional[int]:
        """Parse an optional integer-valued field."""

        numeric_value = PSVIngestor._parse_optional_float(
            value,
            column,
        )

        if numeric_value is None:
            return None

        if not numeric_value.is_integer():
            raise ValueError(
                f"Expected integer value for {column!r}, "
                f"got {value!r}"
            )

        return int(numeric_value)