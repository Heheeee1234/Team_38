from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

DATASETS = {
    "training_setA": DATA_ROOT / "training_setA" / "training_setA",
    "training_setB": DATA_ROOT / "training_setB" / "training_setB",
}

CORE_VITALS = [
    "HR",
    "O2Sat",
    "Resp",
    "SBP",
    "MAP",
    "DBP",
    "Temp",
]

SEPSIS_COLUMN = "SepsisLabel"


# ============================================================
# HELPERS
# ============================================================

def get_missing_runs(series):
    """
    Return lengths of consecutive missing-value runs.

    Example:
        [80, NaN, NaN, 82, NaN]
        -> [2, 1]
    """
    is_missing = series.isna().to_numpy()

    runs = []
    current_run = 0

    for missing in is_missing:
        if missing:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0

    if current_run > 0:
        runs.append(current_run)

    return runs


def percentile(values, p):
    """Safe percentile helper."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(dataset_name, dataset_path):

    print("\n" + "=" * 70)
    print(f"TEMPORAL MISSINGNESS ANALYSIS: {dataset_name}")
    print("=" * 70)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist:")
        print(f"        {dataset_path}")
        return

    files = sorted(dataset_path.glob("*.psv"))

    # Some PhysioNet copies may use files without extension.
    if not files:
        files = sorted(
            p for p in dataset_path.iterdir()
            if p.is_file()
        )

    print(f"Dataset path : {dataset_path}")
    print(f"Patient files: {len(files)}")

    if not files:
        print("[ERROR] No patient files found.")
        return

    # --------------------------------------------------------
    # Global counters
    # --------------------------------------------------------

    total_rows = 0

    total_missing = Counter()
    total_observed = Counter()

    patients_with_missing = Counter()

    # For each vital:
    # list containing each patient's maximum missing run
    max_run_per_patient = {
        vital: []
        for vital in CORE_VITALS
    }

    # All missing run lengths across all patients
    all_runs = {
        vital: []
        for vital in CORE_VITALS
    }

    # Missing runs grouped by length
    run_length_counts = {
        vital: Counter()
        for vital in CORE_VITALS
    }

    # --------------------------------------------------------
    # Sepsis-related missingness
    # --------------------------------------------------------

    sepsis_rows = 0

    missing_at_sepsis = Counter()
    missing_at_nonsepsis = Counter()

    # --------------------------------------------------------
    # Process one patient at a time
    # --------------------------------------------------------

    for index, file_path in enumerate(files, start=1):

        try:
            df = pd.read_csv(
                file_path,
                sep="|",
                low_memory=False
            )
        except Exception as e:
            print(f"[WARNING] Could not read {file_path.name}: {e}")
            continue

        total_rows += len(df)

        # Make sure required columns exist
        missing_columns = [
            col for col in CORE_VITALS + [SEPSIS_COLUMN]
            if col not in df.columns
        ]

        if missing_columns:
            print(
                f"[WARNING] {file_path.name} missing columns: "
                f"{missing_columns}"
            )
            continue

        # ----------------------------------------------------
        # Per-vital analysis
        # ----------------------------------------------------

        for vital in CORE_VITALS:

            series = df[vital]

            missing_count = int(series.isna().sum())
            observed_count = int(series.notna().sum())

            total_missing[vital] += missing_count
            total_observed[vital] += observed_count

            if missing_count > 0:
                patients_with_missing[vital] += 1

            # Consecutive missing runs
            runs = get_missing_runs(series)

            if runs:
                all_runs[vital].extend(runs)
                max_run_per_patient[vital].append(max(runs))

                for run_length in runs:
                    run_length_counts[vital][run_length] += 1
            else:
                # Patient has no missing run.
                max_run_per_patient[vital].append(0)

        # ----------------------------------------------------
        # Missingness around sepsis
        # ----------------------------------------------------

        sepsis_mask = df[SEPSIS_COLUMN] == 1
        nonsepsis_mask = ~sepsis_mask

        sepsis_rows += int(sepsis_mask.sum())

        for vital in CORE_VITALS:

            missing_mask = df[vital].isna()

            missing_at_sepsis[vital] += int(
                (missing_mask & sepsis_mask).sum()
            )

            missing_at_nonsepsis[vital] += int(
                (missing_mask & nonsepsis_mask).sum()
            )

        # Progress
        if index % 1000 == 0:
            print(f"Processed {index}/{len(files)} patients...")

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("-" * 70)
    print("DATASET SUMMARY")
    print("-" * 70)

    print(f"Patient files analyzed : {len(files)}")
    print(f"Total timesteps        : {total_rows:,}")
    print(f"Sepsis-positive rows   : {sepsis_rows:,}")

    # ========================================================
    # PER-VITAL RESULTS
    # ========================================================

    for vital in CORE_VITALS:

        missing = total_missing[vital]
        observed = total_observed[vital]

        total = missing + observed

        missing_rate = (
            100.0 * missing / total
            if total > 0
            else 0.0
        )

        patient_missing_rate = (
            100.0 * patients_with_missing[vital] / len(files)
            if files
            else 0.0
        )

        runs = all_runs[vital]
        patient_max_runs = max_run_per_patient[vital]

        print("\n" + "-" * 70)
        print(vital)
        print("-" * 70)

        print(f"Observed values          : {observed:,}")
        print(f"Missing values           : {missing:,}")
        print(f"Overall missing rate     : {missing_rate:.2f}%")

        print(
            f"Patients with missing   : "
            f"{patients_with_missing[vital]:,} "
            f"({patient_missing_rate:.2f}%)"
        )

        if runs:

            print("\nConsecutive missing runs:")

            print(
                f"  Total runs             : "
                f"{len(runs):,}"
            )

            print(
                f"  Mean run length       : "
                f"{np.mean(runs):.2f}"
            )

            print(
                f"  Median run length     : "
                f"{np.median(runs):.2f}"
            )

            print(
                f"  95th percentile       : "
                f"{percentile(runs, 95):.2f}"
            )

            print(
                f"  Maximum run           : "
                f"{max(runs)}"
            )

            print("\nMissing-run distribution:")

            # Show runs 1 through 5
            for length in range(1, 6):
                count = run_length_counts[vital][length]

                print(
                    f"  {length} timestep"
                    f"{'s' if length != 1 else ' '} "
                    f"             : {count:,}"
                )

            longer = sum(
                count
                for length, count in run_length_counts[vital].items()
                if length >= 6
            )

            print(
                f"  6+ timesteps          : {longer:,}"
            )

            print("\nMaximum missing run per patient:")

            print(
                f"  Mean                  : "
                f"{np.mean(patient_max_runs):.2f}"
            )

            print(
                f"  Median                : "
                f"{np.median(patient_max_runs):.2f}"
            )

            print(
                f"  95th percentile      : "
                f"{percentile(patient_max_runs, 95):.2f}"
            )

            print(
                f"  Maximum              : "
                f"{max(patient_max_runs)}"
            )

        else:
            print("\nNo missing values found.")

        # ----------------------------------------------------
        # Sepsis-related missingness
        # ----------------------------------------------------

        if sepsis_rows > 0:

            sepsis_missing_rate = (
                100.0 *
                missing_at_sepsis[vital] /
                sepsis_rows
            )

            nonsepsis_rows = total_rows - sepsis_rows

            nonsepsis_missing_rate = (
                100.0 *
                missing_at_nonsepsis[vital] /
                nonsepsis_rows
                if nonsepsis_rows > 0
                else 0.0
            )

            print("\nMissingness by SepsisLabel:")

            print(
                f"  At SepsisLabel=1      : "
                f"{sepsis_missing_rate:.2f}%"
            )

            print(
                f"  At SepsisLabel=0      : "
                f"{nonsepsis_missing_rate:.2f}%"
            )

    # ========================================================
    # COMPACT SUMMARY TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("COMPACT SUMMARY")
    print("=" * 70)

    print(
        f"{'Vital':<10}"
        f"{'Missing %':>12}"
        f"{'Median Max':>15}"
        f"{'95% Max':>12}"
        f"{'Max':>10}"
    )

    print("-" * 70)

    for vital in CORE_VITALS:

        missing = total_missing[vital]
        observed = total_observed[vital]
        total = missing + observed

        missing_rate = (
            100.0 * missing / total
            if total > 0
            else 0.0
        )

        patient_max_runs = max_run_per_patient[vital]

        print(
            f"{vital:<10}"
            f"{missing_rate:>11.2f}%"
            f"{np.median(patient_max_runs):>15.2f}"
            f"{percentile(patient_max_runs, 95):>12.2f}"
            f"{max(patient_max_runs):>10}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CORE VITAL TEMPORAL MISSINGNESS ANALYSIS")
    print("=" * 70)

    print("\nCore vitals:")
    print(", ".join(CORE_VITALS))

    for dataset_name, dataset_path in DATASETS.items():
        analyze_dataset(dataset_name, dataset_path)


if __name__ == "__main__":
    main()