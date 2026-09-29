"""Adapters for converting ingested PSV rows into evaluation streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from clinical_data_gen.evaluation.runner import OnlineEvaluationRunner
from clinical_data_gen.ingestion.psv_ingestor import PSVIngestor
from clinical_data_gen.state.patient_state import PatientState
from clinical_data_gen.state.patient_state_manager import (
    PatientStateManager,
)


@dataclass(frozen=True)
class EvaluationSample:
    """One labelled observation for online evaluation."""

    state: PatientState
    label: int


class PSVEvaluationStream:
    """
    Convert a PSV file into an ordered PatientState/label stream.

    SepsisLabel is used here strictly as the offline evaluation target.
    It is not stored inside PatientState and is not treated as
    clinician feedback.
    """

    def __init__(
        self,
        psv_path: str | Path,
        *,
        patient_id: str | None = None,
    ) -> None:
        self.psv_path = Path(psv_path)
        self.patient_id = patient_id or self.psv_path.stem

    def samples(self) -> Iterator[tuple[PatientState, int]]:
        """
        Yield PatientState and SepsisLabel pairs in PSV order.
        """

        state_manager = PatientStateManager()
        ingestor = PSVIngestor(state_manager)

        rows = self._read_rows()

        for row in rows:
            state, label = ingestor.ingest_row(
                patient_id=self.patient_id,
                row=row,
            )

            yield state, int(label)

    def _read_rows(self) -> Iterator[dict[str, str | None]]:
        """
        Read the PSV file while preserving the existing ingestion contract.

        The PSV ingestor remains responsible for schema validation and
        clinical-value parsing.
        """

        with self.psv_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            header_line = file.readline().strip()

            if not header_line:
                raise ValueError(
                    f"Empty PSV file: {self.psv_path}"
                )

            columns = header_line.split("|")

            for line_number, line in enumerate(
                file,
                start=2,
            ):
                line = line.rstrip("\n\r")

                if not line:
                    continue

                values = line.split("|")

                if len(values) != len(columns):
                    raise ValueError(
                        f"Malformed PSV row at line "
                        f"{line_number}: expected "
                        f"{len(columns)} columns, got "
                        f"{len(values)}"
                    )

                yield {
                    column: (
                        value
                        if value != ""
                        else None
                    )
                    for column, value in zip(
                        columns,
                        values,
                    )
                }


def evaluate_psv_file(
    psv_path: str | Path,
    runner: OnlineEvaluationRunner,
) -> None:
    """
    Evaluate one PSV patient stream using an existing runner.
    """

    stream = PSVEvaluationStream(psv_path)

    runner.evaluate(
        stream.samples()
    )


def evaluate_psv_files(
    psv_paths: Iterable[str | Path],
    runner: OnlineEvaluationRunner,
) -> None:
    """
    Evaluate multiple patient PSV files sequentially.

    Model state is intentionally preserved between patients, allowing
    the online learner to learn continuously from the stream.
    """

    for psv_path in psv_paths:
        evaluate_psv_file(
            psv_path,
            runner,
        )

def discover_psv_files(
    input_path: str | Path,
) -> list[Path]:
    """Find PSV files from a file or directory."""

    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() != ".psv":
            raise ValueError(
                f"Input file is not a PSV file: {input_path}"
            )

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    files = sorted(input_path.glob("*.psv"))

    if not files:
        raise FileNotFoundError(
            f"No .psv files found in {input_path}"
        )

    return files