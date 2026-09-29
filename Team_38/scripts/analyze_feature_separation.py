"""
Analyze whether the current physiological features distinguish
positive and negative observations.

This is an analysis-only script. It does NOT train or modify the model.

Outputs:
    feature_separation.csv
    feature_separation.png
    top_feature_distributions.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CORE_VITALS = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
)

# Same physiological centers/scales currently used by
# PatientFeatureExtractor.
SCALING = {
    "HR": (80.0, 30.0),
    "O2Sat": (97.0, 5.0),
    "Temp": (37.0, 1.0),
    "SBP": (120.0, 30.0),
    "MAP": (80.0, 20.0),
    "DBP": (70.0, 20.0),
    "Resp": (18.0, 8.0),
}


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def read_patient_file(path: Path):
    """
    Yield rows from one PSV patient file.

    Returns:
        (patient_id, row_dict)
    """
    patient_id = path.name

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="|",
        )

        for row in reader:
            yield patient_id, row


def discover_patient_files(
    input_path: Path,
    max_patients: int | None,
):
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.psv"))

    if not files:
        raise FileNotFoundError(
            f"No PSV files found under {input_path}"
        )

    if max_patients is not None:
        if max_patients <= 0:
            raise ValueError(
                "max_patients must be positive"
            )

        files = files[:max_patients]

    return files


def safe_mean(values):
    if not values:
        return float("nan")

    return float(np.mean(values))


def safe_std(values):
    if len(values) < 2:
        return float("nan")

    return float(np.std(values, ddof=1))


def safe_median(values):
    if not values:
        return float("nan")

    return float(np.median(values))


def standardized_mean_difference(
    positive,
    negative,
):
    """
    Standardized mean difference.

    Roughly:
        difference in means
        -------------------
        pooled standard deviation

    Larger absolute values mean stronger separation.
    """
    if not positive or not negative:
        return float("nan")

    positive_std = np.std(positive, ddof=1) if len(positive) > 1 else 0
    negative_std = np.std(negative, ddof=1) if len(negative) > 1 else 0

    pooled_variance = (
        (positive_std**2 + negative_std**2) / 2.0
    )

    if pooled_variance <= 0:
        return 0.0

    return (
        (np.mean(positive) - np.mean(negative))
        / math.sqrt(pooled_variance)
    )


def process_dataset(
    input_path: Path,
    max_patients: int | None,
    max_forward_fill: int,
):
    """
    Reconstruct the core current-value and delta features.

    The forward-fill behavior matches the current preprocessing policy:
    HR, O2Sat, Resp, SBP, MAP and DBP may be carried forward for a
    short gap. Temperature is not forward-filled.
    """

    files = discover_patient_files(
        input_path,
        max_patients,
    )

    feature_values = defaultdict(
        lambda: {
            0: [],
            1: [],
        }
    )

    missing_counts = defaultdict(
        lambda: {
            0: 0,
            1: 0,
        }
    )

    total_counts = {
        0: 0,
        1: 0,
    }

    patients_with_positive = 0

    for file_index, path in enumerate(files, start=1):
        previous_values = {
            vital: None
            for vital in CORE_VITALS
        }

        last_observed = {
            vital: None
            for vital in CORE_VITALS
        }

        forward_fill_count = {
            vital: 0
            for vital in CORE_VITALS
        }

        patient_has_positive = False

        for _, row in read_patient_file(path):
            label_value = parse_float(
                row.get("SepsisLabel")
            )

            if label_value is None:
                continue

            label = int(label_value)

            if label not in (0, 1):
                continue

            total_counts[label] += 1

            if label == 1:
                patient_has_positive = True

            for vital in CORE_VITALS:
                raw_value = parse_float(
                    row.get(vital)
                )

                # --------------------------------------------------
                # Missingness
                # --------------------------------------------------

                if raw_value is None:
                    missing_counts[vital][label] += 1

                    if vital in {
                        "HR",
                        "O2Sat",
                        "Resp",
                        "SBP",
                        "MAP",
                        "DBP",
                    }:
                        if (
                            last_observed[vital] is not None
                            and forward_fill_count[vital]
                            < max_forward_fill
                        ):
                            current_value = last_observed[vital]
                            forward_fill_count[vital] += 1
                        else:
                            current_value = None
                    else:
                        # Temperature is intentionally not
                        # forward-filled.
                        current_value = None
                else:
                    current_value = raw_value
                    last_observed[vital] = raw_value
                    forward_fill_count[vital] = 0

                # --------------------------------------------------
                # Current value feature
                # --------------------------------------------------

                if current_value is not None:
                    center, scale = SCALING[vital]

                    normalized = (
                        current_value - center
                    ) / scale

                    feature_values[
                        f"{vital}_current"
                    ][label].append(normalized)

                # --------------------------------------------------
                # Delta feature
                # --------------------------------------------------

                previous = previous_values[vital]

                if (
                    current_value is not None
                    and previous is not None
                ):
                    center, scale = SCALING[vital]

                    delta = (
                        current_value - previous
                    ) / scale

                    feature_values[
                        f"{vital}_delta"
                    ][label].append(delta)

                previous_values[vital] = current_value

        if patient_has_positive:
            patients_with_positive += 1

        if file_index % 100 == 0:
            print(
                f"Processed "
                f"{file_index:,}/{len(files):,} patients..."
            )

    return (
        feature_values,
        missing_counts,
        total_counts,
        patients_with_positive,
        len(files),
    )


def build_statistics(
    feature_values,
    missing_counts,
    total_counts,
):
    rows = []

    for feature_name, values_by_label in feature_values.items():
        negative = values_by_label[0]
        positive = values_by_label[1]

        if not positive or not negative:
            continue

        negative_mean = safe_mean(negative)
        positive_mean = safe_mean(positive)

        smd = standardized_mean_difference(
            positive,
            negative,
        )

        rows.append(
            {
                "feature": feature_name,
                "negative_count": len(negative),
                "positive_count": len(positive),
                "negative_mean": negative_mean,
                "positive_mean": positive_mean,
                "mean_difference": (
                    positive_mean - negative_mean
                ),
                "negative_median": safe_median(negative),
                "positive_median": safe_median(positive),
                "negative_std": safe_std(negative),
                "positive_std": safe_std(positive),
                "absolute_smd": abs(smd),
                "smd": smd,
            }
        )

    # Missingness features
    for vital in CORE_VITALS:
        for label in (0, 1):
            pass

        negative_total = total_counts[0]
        positive_total = total_counts[1]

        negative_missing = missing_counts[vital][0]
        positive_missing = missing_counts[vital][1]

        negative_rate = (
            negative_missing / negative_total
            if negative_total
            else float("nan")
        )

        positive_rate = (
            positive_missing / positive_total
            if positive_total
            else float("nan")
        )

        rows.append(
            {
                "feature": f"{vital}_missing_rate",
                "negative_count": negative_total,
                "positive_count": positive_total,
                "negative_mean": negative_rate,
                "positive_mean": positive_rate,
                "mean_difference": (
                    positive_rate - negative_rate
                ),
                "negative_median": negative_rate,
                "positive_median": positive_rate,
                "negative_std": float("nan"),
                "positive_std": float("nan"),
                "absolute_smd": abs(
                    positive_rate - negative_rate
                ),
                "smd": (
                    positive_rate - negative_rate
                ),
            }
        )

    rows.sort(
        key=lambda row: row["absolute_smd"],
        reverse=True,
    )

    return rows


def save_csv(rows, path: Path):
    fieldnames = [
        "feature",
        "negative_count",
        "positive_count",
        "negative_mean",
        "positive_mean",
        "mean_difference",
        "negative_median",
        "positive_median",
        "negative_std",
        "positive_std",
        "absolute_smd",
        "smd",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def plot_feature_separation(
    rows,
    output: Path,
    top_n: int = 12,
):
    selected = rows[:top_n]

    selected = list(reversed(selected))

    labels = [
        row["feature"]
        for row in selected
    ]

    values = [
        row["smd"]
        for row in selected
    ]

    plt.figure(figsize=(10, 7))

    plt.barh(
        labels,
        values,
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )

    plt.xlabel(
        "Standardized mean difference\n"
        "(positive vs negative)"
    )

    plt.title(
        "Feature Separation: Positive vs Negative"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()
    plt.savefig(
        output,
        dpi=180,
    )
    plt.close()


def plot_top_feature_distributions(
    feature_values,
    rows,
    output: Path,
    top_n: int = 4,
):
    selected = [
        row["feature"]
        for row in rows
        if not row["feature"].endswith("_missing_rate")
    ][:top_n]

    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(10, 3 * len(selected)),
    )

    if len(selected) == 1:
        axes = [axes]

    for axis, feature_name in zip(
        axes,
        selected,
    ):
        negative = feature_values[feature_name][0]
        positive = feature_values[feature_name][1]

        all_values = negative + positive

        if not all_values:
            continue

        lower = np.percentile(
            all_values,
            1,
        )

        upper = np.percentile(
            all_values,
            99,
        )

        if lower == upper:
            lower -= 1
            upper += 1

        bins = np.linspace(
            lower,
            upper,
            35,
        )

        axis.hist(
            negative,
            bins=bins,
            alpha=0.60,
            density=True,
            label="Negative",
        )

        axis.hist(
            positive,
            bins=bins,
            alpha=0.60,
            density=True,
            label="Positive",
        )

        axis.set_title(feature_name)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25)

        axis.legend()

    axes[-1].set_xlabel(
        "Feature value"
    )

    fig.suptitle(
        "Top Feature Distributions",
        fontsize=15,
    )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_summary(
    rows,
    total_counts,
    patients_with_positive,
    patient_count,
):
    print()
    print("=" * 78)
    print("FEATURE SEPARATION ANALYSIS")
    print("=" * 78)

    print(
        f"Patients analyzed : {patient_count:,}"
    )

    print(
        f"Negative samples  : {total_counts[0]:,}"
    )

    print(
        f"Positive samples  : {total_counts[1]:,}"
    )

    print(
        f"Patients with positive label: "
        f"{patients_with_positive:,}"
    )

    print()
    print(
        "Top features by absolute standardized mean difference:"
    )

    print()
    print(
        f"{'Feature':<25}"
        f"{'Negative mean':>16}"
        f"{'Positive mean':>16}"
        f"{'SMD':>12}"
    )

    print("-" * 69)

    for row in rows[:15]:
        print(
            f"{row['feature']:<25}"
            f"{row['negative_mean']:>16.4f}"
            f"{row['positive_mean']:>16.4f}"
            f"{row['smd']:>12.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze separation between positive and "
            "negative observations using the current "
            "physiological feature representation."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/training_setA/training_setA"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/feature_analysis"
        ),
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-forward-fill",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("CURRENT FEATURE SEPARATION ANALYSIS")
    print("=" * 78)

    print(
        f"Input          : {args.input}"
    )

    print(
        f"Patients       : "
        f"{args.max_patients or 'all'}"
    )

    print(
        f"Forward fill   : "
        f"{args.max_forward_fill}"
    )

    (
        feature_values,
        missing_counts,
        total_counts,
        patients_with_positive,
        patient_count,
    ) = process_dataset(
        input_path=args.input,
        max_patients=args.max_patients,
        max_forward_fill=args.max_forward_fill,
    )

    rows = build_statistics(
        feature_values,
        missing_counts,
        total_counts,
    )

    csv_path = (
        args.output / "feature_separation.csv"
    )

    separation_plot = (
        args.output / "feature_separation.png"
    )

    distributions_plot = (
        args.output
        / "top_feature_distributions.png"
    )

    save_csv(
        rows,
        csv_path,
    )

    plot_feature_separation(
        rows,
        separation_plot,
    )

    plot_top_feature_distributions(
        feature_values,
        rows,
        distributions_plot,
    )

    print_summary(
        rows,
        total_counts,
        patients_with_positive,
        patient_count,
    )

    print()
    print("Generated:")
    print(f"  {csv_path}")
    print(f"  {separation_plot}")
    print(f"  {distributions_plot}")


if __name__ == "__main__":
    main()