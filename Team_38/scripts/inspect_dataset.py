from pathlib import Path
import pandas as pd
import glob

PROJECT_ROOT = Path(__file__).parents[1]

print(f"Root of project: {PROJECT_ROOT}")

DATASETS = [
    PROJECT_ROOT / "data" / "training_setA" / "training_setA",
    PROJECT_ROOT / "data" / "training_setB" / "training_setB"
]

# List containing all the columns that are expected in a PSV file

EXPECTED_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel"
]

# Analyzing each dataset
def inspect_dataset(dataset_path: Path):
    print(f"Checking for the dataset path: {dataset_path}")
    print(f"Dataset: {dataset_path.name}")

    files = sorted(dataset_path.glob("*.psv"))

    print(f"Patient files found: {len(files)}")

    if not files:
        print(f"No patient files found in {len(files)}")
        return

    total_rows = 0
    total_sepsis_rows = 0
    patients_with_sepsis = 0

    row_counts = []

    missing_counts = pd.Series(
        0,
        index=EXPECTED_COLUMNS,
        dtype="int64"
    )

    # Number of patients with missing value
    patients_with_missing = pd.Series(
        0,
        index=EXPECTED_COLUMNS,
        dtype="int64"
    )

    schema_mismatches = []

    icu_durations = []

    non_monotonic_iculos = 0
    duplicate_iculos = 0

    ages = []
    genders = []
    units = []

    duplicate_rows = 0
    duplicate_rows = 0

    # --------------------------------------------------------
    # Read patient files
    # --------------------------------------------------------

    for file_path in files:

        df = pd.read_csv(
            file_path,
            sep="|"
        )

        patient_id = file_path.stem

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        if list(df.columns) != EXPECTED_COLUMNS:
            schema_mismatches.append(patient_id)

        # ----------------------------------------------------
        # Number of rows
        # ----------------------------------------------------

        n_rows = len(df)

        total_rows += n_rows
        row_counts.append(n_rows)

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        for column in EXPECTED_COLUMNS:

            missing = df[column].isna().sum()

            missing_counts[column] += missing

            if missing > 0:
                patients_with_missing[column] += 1

        # ----------------------------------------------------
        # Sepsis
        # ----------------------------------------------------

        sepsis = pd.to_numeric(
            df["SepsisLabel"],
            errors="coerce"
        ).fillna(0)

        positive_rows = int((sepsis == 1).sum())

        total_sepsis_rows += positive_rows

        if positive_rows > 0:
            patients_with_sepsis += 1

        # ----------------------------------------------------
        # ICU timeline
        # ----------------------------------------------------

        icu = pd.to_numeric(
            df["ICULOS"],
            errors="coerce"
        ).dropna()

        if len(icu) > 0:

            icu_durations.append(
                icu.max() - icu.min()
            )

            if not icu.is_monotonic_increasing:
                non_monotonic_iculos += 1

            if icu.duplicated().any():
                duplicate_iculos += 1

        # ----------------------------------------------------
        # Static information
        # ----------------------------------------------------

        age = pd.to_numeric(
            df["Age"],
            errors="coerce"
        ).dropna()

        if len(age):
            ages.append(age.iloc[0])

        gender_values = df["Gender"].dropna()

        if len(gender_values):
            genders.append(gender_values.iloc[0])

        # Unit1 / Unit2 are usually binary indicators.
        for column in ["Unit1", "Unit2"]:
            values = df[column].dropna()

            if len(values):
                units.extend(
                    [(column, value) for value in values.unique()]
                )

        # ----------------------------------------------------
        # Duplicate rows
        # ----------------------------------------------------

        duplicate_rows += df.duplicated().sum()

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n--- BASIC DATASET INFORMATION ---")

    print(f"Patient files       : {len(files):,}")
    print(f"Total timesteps     : {total_rows:,}")

    if row_counts:

        print(
            f"Timesteps / patient : "
            f"min={min(row_counts):,}, "
            f"median={int(pd.Series(row_counts).median()):,}, "
            f"mean={pd.Series(row_counts).mean():.1f}, "
            f"max={max(row_counts):,}"
        )

    # ========================================================
    # SCHEMA
    # ========================================================

    print("\n--- SCHEMA ---")

    if not schema_mismatches:

        print("All files have the expected schema: YES")

    else:

        print("All files have the expected schema: NO")
        print(
            f"Files with schema mismatch: "
            f"{len(schema_mismatches)}"
        )

        print(
            "First mismatches:",
            schema_mismatches[:10]
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print("\n--- MISSING VALUES ---")

    missing_report = pd.DataFrame({
        "missing": missing_counts,
        "missing_%": (
            missing_counts / total_rows * 100
        ),
        "patients_affected": patients_with_missing,
        "patients_affected_%": (
            patients_with_missing
            / len(files)
            * 100
        ),
    })

    missing_report = missing_report.sort_values(
        "missing_%",
        ascending=False
    )

    print(
        missing_report.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # ========================================================
    # SEPSIS
    # ========================================================

    print("\n--- SEPSIS LABEL ---")

    print(
        f"Positive timesteps      : "
        f"{total_sepsis_rows:,}"
    )

    print(
        f"Positive timestep rate  : "
        f"{total_sepsis_rows / total_rows * 100:.4f}%"
    )

    print(
        f"Patients with sepsis    : "
        f"{patients_with_sepsis:,}"
    )

    print(
        f"Patient sepsis rate     : "
        f"{patients_with_sepsis / len(files) * 100:.2f}%"
    )

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    print("\n--- PATIENT INFORMATION ---")

    if ages:

        age_series = pd.Series(ages)

        print(
            f"Age: "
            f"min={age_series.min():.1f}, "
            f"median={age_series.median():.1f}, "
            f"mean={age_series.mean():.1f}, "
            f"max={age_series.max():.1f}"
        )

    if genders:

        print("\nGender distribution:")

        print(
            pd.Series(genders)
            .value_counts()
            .to_string()
        )

    # ========================================================
    # TEMPORAL INFORMATION
    # ========================================================

    print("\n--- TEMPORAL INFORMATION ---")

    if icu_durations:

        duration_series = pd.Series(
            icu_durations
        )

        print(
            f"ICU duration (hours): "
            f"min={duration_series.min():.1f}, "
            f"median={duration_series.median():.1f}, "
            f"mean={duration_series.mean():.1f}, "
            f"max={duration_series.max():.1f}"
        )

    print(
        f"Non-monotonic ICULOS files : "
        f"{non_monotonic_iculos}"
    )

    print(
        f"Files with duplicate ICULOS: "
        f"{duplicate_iculos}"
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print("\n--- BASIC DATA QUALITY ---")

    print(
        f"Duplicate rows across dataset : "
        f"{duplicate_rows:,}"
    )

    # ========================================================
    # MOST / LEAST MISSING
    # ========================================================

    print("\n--- MOST MISSING VARIABLES ---")

    print(
        missing_report[
            ["missing_%", "patients_affected_%"]
        ]
        .head(10)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\n--- LEAST MISSING VARIABLES ---")

    print(
        missing_report[
            ["missing_%", "patients_affected_%"]
        ]
        .tail(10)
        .sort_values("missing_%")
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLINICAL DATASET INSPECTION")
    print("=" * 70)

    for dataset_path in DATASETS:

        if not dataset_path.exists():

            print(
                f"\nWARNING: "
                f"{dataset_path} does not exist."
            )

            continue

        inspect_dataset(dataset_path)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()