"""
Compact diagnostic analysis for Online Logistic Regression.

Reads an existing predictions.jsonl file. No model retraining.

Outputs:
    probability_distribution.png
    precision_recall_thresholds.png
    model_diagnostics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_predictions(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError("No predictions found.")

    required = {
        "probability",
        "true_label",
    }

    missing = required - records[0].keys()

    if missing:
        raise ValueError(
            f"predictions.jsonl is missing fields: {missing}"
        )

    probabilities = np.asarray(
        [float(r["probability"]) for r in records],
        dtype=float,
    )

    labels = np.asarray(
        [int(r["true_label"]) for r in records],
        dtype=int,
    )

    return probabilities, labels


def confusion_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
):
    predictions = probabilities >= threshold

    tp = int(np.sum(
        (predictions == 1) & (labels == 1)
    ))

    fp = int(np.sum(
        (predictions == 1) & (labels == 0)
    ))

    tn = int(np.sum(
        (predictions == 0) & (labels == 0)
    ))

    fn = int(np.sum(
        (predictions == 0) & (labels == 1)
    ))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    alert_rate = float(np.mean(predictions))

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "alert_rate": alert_rate,
    }


def threshold_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
):
    # More useful thresholds for our probability range.
    thresholds = np.concatenate([
        np.arange(0.01, 0.10, 0.01),
        np.arange(0.10, 0.31, 0.01),
        np.arange(0.35, 0.51, 0.05),
    ])

    results = []

    for threshold in thresholds:
        results.append(
            confusion_metrics(
                labels,
                probabilities,
                float(threshold),
            )
        )

    return results


def precision_recall_points(
    labels: np.ndarray,
    probabilities: np.ndarray,
):
    thresholds = np.unique(probabilities)
    thresholds = np.sort(thresholds)[::-1]

    positive_count = np.sum(labels == 1)

    precisions = []
    recalls = []
    used_thresholds = []

    for threshold in thresholds:
        predictions = probabilities >= threshold

        tp = np.sum(
            (predictions == 1) & (labels == 1)
        )

        fp = np.sum(
            (predictions == 1) & (labels == 0)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 1.0
        )

        recall = (
            tp / positive_count
            if positive_count > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        used_thresholds.append(threshold)

    return (
        np.asarray(recalls),
        np.asarray(precisions),
        np.asarray(used_thresholds),
    )


def print_probability_statistics(
    probabilities,
    labels,
):
    print()
    print("=" * 78)
    print("PROBABILITY DISTRIBUTION")
    print("=" * 78)

    for label, name in [
        (0, "NEGATIVE"),
        (1, "POSITIVE"),
    ]:
        values = probabilities[labels == label]

        percentiles = np.percentile(
            values,
            [1, 5, 25, 50, 75, 95, 99],
        )

        print()
        print(name)
        print(f"  Count  : {len(values):,}")
        print(f"  Mean   : {np.mean(values):.6f}")
        print(f"  Median : {np.median(values):.6f}")
        print(f"  Min    : {np.min(values):.6f}")
        print(f"  Max    : {np.max(values):.6f}")
        print(
            "  P01/P05/P25/P50/P75/P95/P99:"
        )
        print(
            "  "
            + " / ".join(
                f"{x:.4f}"
                for x in percentiles
            )
        )


def print_threshold_table(results):
    print()
    print("=" * 78)
    print("THRESHOLD ANALYSIS")
    print("=" * 78)

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Alert Rate':>12}"
        f"{'TP':>7}"
        f"{'FP':>7}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{result['threshold']:>10.2f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['alert_rate']:>12.4f}"
            f"{result['tp']:>7}"
            f"{result['fp']:>7}"
        )

    best = max(
        results,
        key=lambda x: x["f1"],
    )

    print()
    print("Best F1 in tested thresholds:")
    print(
        f"  Threshold : {best['threshold']:.2f}"
    )
    print(
        f"  Precision : {best['precision']:.4f}"
    )
    print(
        f"  Recall    : {best['recall']:.4f}"
    )
    print(
        f"  F1        : {best['f1']:.4f}"
    )
    print(
        f"  Alert rate: {best['alert_rate']:.4f}"
    )


def plot_probability_distribution(
    probabilities,
    labels,
    output,
):
    negative = probabilities[labels == 0]
    positive = probabilities[labels == 1]

    plt.figure(figsize=(10, 6))

    bins = np.linspace(
        0,
        min(1.0, max(probabilities.max() * 1.05, 0.2)),
        50,
    )

    plt.hist(
        negative,
        bins=bins,
        density=True,
        alpha=0.60,
        label=f"Negative (n={len(negative):,})",
    )

    plt.hist(
        positive,
        bins=bins,
        density=True,
        alpha=0.60,
        label=f"Positive (n={len(positive):,})",
    )

    plt.axvline(
        0.5,
        linestyle="--",
        linewidth=2,
        label="Current threshold = 0.5",
    )

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(
        "Predicted Probability: Positive vs Negative"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_precision_recall(
    labels,
    probabilities,
    threshold_results,
    output,
):
    recall, precision, thresholds = (
        precision_recall_points(
            labels,
            probabilities,
        )
    )

    order = np.argsort(recall)

    recall = recall[order]
    precision = precision[order]

    prevalence = np.mean(labels)

    plt.figure(figsize=(9, 6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label="Online LR",
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.5,
        label=f"Prevalence = {prevalence:.4f}",
    )

    # Mark the thresholds we explicitly tested.
    threshold_lookup = {
        round(x["threshold"], 2): x
        for x in threshold_results
    }

    for threshold, result in threshold_lookup.items():
        plt.scatter(
            result["recall"],
            result["precision"],
            s=25,
        )

    best = max(
        threshold_results,
        key=lambda x: x["f1"],
    )

    plt.scatter(
        best["recall"],
        best["precision"],
        s=90,
        marker="*",
        label=(
            f"Best tested F1={best['f1']:.3f} "
            f"@ threshold={best['threshold']:.2f}"
        ),
        zorder=5,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Precision–Recall Curve with Threshold Analysis"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_csv(results, output):
    fieldnames = [
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "specificity",
        "alert_rate",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to predictions.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/model_diagnostics"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {args.input}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading predictions from: {args.input}"
    )

    probabilities, labels = load_predictions(
        args.input
    )

    print(
        f"Observations: {len(labels):,}"
    )
    print(
        f"Positive:     {np.sum(labels == 1):,}"
    )
    print(
        f"Negative:     {np.sum(labels == 0):,}"
    )

    print_probability_statistics(
        probabilities,
        labels,
    )

    results = threshold_analysis(
        labels,
        probabilities,
    )

    print_threshold_table(results)

    save_csv(
        results,
        args.output / "model_diagnostics.csv",
    )

    plot_probability_distribution(
        probabilities,
        labels,
        args.output
        / "probability_distribution.png",
    )

    plot_precision_recall(
        labels,
        probabilities,
        results,
        args.output
        / "precision_recall_thresholds.png",
    )

    print()
    print("Generated:")
    print(
        f"  {args.output / 'model_diagnostics.csv'}"
    )
    print(
        f"  {args.output / 'probability_distribution.png'}"
    )
    print(
        f"  {args.output / 'precision_recall_thresholds.png'}"
    )


if __name__ == "__main__":
    main()