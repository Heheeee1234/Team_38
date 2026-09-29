"""
Generate diagnostic plots from Online Logistic Regression predictions.

Input:
    data/evaluation/online_lr/predictions.jsonl

Output:
    data/evaluation/online_lr/diagnostics/
        probability_distribution.png
        precision_recall_curve.png
        threshold_metrics.png
        rolling_performance.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_predictions(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            required = {
                "probability",
                "predicted_label",
                "true_label",
            }

            missing = required - record.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {missing}"
                )

            records.append(record)

    if not records:
        raise ValueError("No prediction records found.")

    return records


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)

    x_sorted = x[order]
    y_sorted = y[order]

    return float(np.trapezoid(y_sorted, x_sorted))


def precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    thresholds = np.unique(probabilities)
    thresholds = np.sort(thresholds)[::-1]

    precisions = []
    recalls = []

    positive_count = np.sum(y_true == 1)

    for threshold in thresholds:
        predictions = probabilities >= threshold

        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))

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

    return (
        np.asarray(recalls),
        np.asarray(precisions),
        thresholds,
    )


def threshold_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    thresholds = np.linspace(0.01, 0.99, 99)

    precision_values = []
    recall_values = []
    f1_values = []
    alert_rates = []

    total = len(y_true)

    for threshold in thresholds:
        predictions = probabilities >= threshold

        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))

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

        alert_rate = np.sum(predictions) / total

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        alert_rates.append(alert_rate)

    return (
        thresholds,
        np.asarray(precision_values),
        np.asarray(recall_values),
        np.asarray(f1_values),
        np.asarray(alert_rates),
    )


def rolling_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    window: int = 2000,
):
    """
    Calculate rolling PR-AUC and log loss.

    This is intentionally done only for plotting and does not modify
    the model or evaluation results.
    """

    centers = []
    pr_auc_values = []
    log_loss_values = []

    for end in range(window, len(y_true) + 1, window):
        start = end - window

        y_window = y_true[start:end]
        p_window = probabilities[start:end]

        # PR-AUC needs both classes.
        if np.any(y_window == 1) and np.any(y_window == 0):
            recall, precision, _ = precision_recall_curve(
                y_window,
                p_window,
            )

            # Recall is normally generated in descending-threshold
            # order, so sort before integration.
            order = np.argsort(recall)

            pr_auc = auc_trapezoid(
                recall[order],
                precision[order],
            )
        else:
            pr_auc = np.nan

        p_safe = np.clip(p_window, 1e-15, 1 - 1e-15)

        loss = -np.mean(
            y_window * np.log(p_safe)
            + (1 - y_window) * np.log(1 - p_safe)
        )

        centers.append(end)
        pr_auc_values.append(pr_auc)
        log_loss_values.append(loss)

    return (
        np.asarray(centers),
        np.asarray(pr_auc_values),
        np.asarray(log_loss_values),
    )


def plot_probability_distribution(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    output: Path,
):
    plt.figure(figsize=(10, 6))

    negative = probabilities[y_true == 0]
    positive = probabilities[y_true == 1]

    bins = np.linspace(0, 1, 51)

    plt.hist(
        negative,
        bins=bins,
        alpha=0.65,
        density=True,
        label=f"Negative (n={len(negative):,})",
    )

    plt.hist(
        positive,
        bins=bins,
        alpha=0.65,
        density=True,
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
    plt.title("Online Logistic Regression: Predicted Probability Distribution")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_precision_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output: Path,
):
    recall, precision, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    order = np.argsort(recall)

    recall = recall[order]
    precision = precision[order]

    prevalence = np.mean(y_true)

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label="Online Logistic Regression",
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        linewidth=2,
        label=f"Positive prevalence = {prevalence:.4f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output: Path,
):
    (
        thresholds,
        precision,
        recall,
        f1,
        alert_rate,
    ) = threshold_analysis(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        precision,
        linewidth=2,
        label="Precision",
    )

    plt.plot(
        thresholds,
        recall,
        linewidth=2,
        label="Recall",
    )

    plt.plot(
        thresholds,
        f1,
        linewidth=2,
        label="F1",
    )

    plt.axvline(
        0.5,
        linestyle="--",
        linewidth=2,
        label="Current threshold = 0.5",
    )

    best_index = np.argmax(f1)

    plt.scatter(
        thresholds[best_index],
        f1[best_index],
        s=60,
        zorder=5,
        label=(
            f"Best F1={f1[best_index]:.3f} "
            f"at threshold={thresholds[best_index]:.2f}"
        ),
    )

    plt.xlabel("Classification threshold")
    plt.ylabel("Metric")
    plt.title("Threshold Sensitivity")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()

    print(
        "\nThreshold analysis:"
    )
    print(
        f"  Best F1       : {f1[best_index]:.4f}"
    )
    print(
        f"  Best threshold: {thresholds[best_index]:.2f}"
    )
    print(
        f"  Precision     : {precision[best_index]:.4f}"
    )
    print(
        f"  Recall        : {recall[best_index]:.4f}"
    )
    print(
        f"  Alert rate    : {alert_rate[best_index]:.4f}"
    )


def plot_rolling_performance(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output: Path,
    window: int,
):
    (
        centers,
        pr_auc,
        log_loss,
    ) = rolling_metrics(
        y_true,
        probabilities,
        window=window,
    )

    if len(centers) == 0:
        print(
            "Not enough observations for rolling-performance plot."
        )
        return

    figure, axis1 = plt.subplots(figsize=(10, 6))

    axis1.plot(
        centers,
        pr_auc,
        linewidth=2,
        label="Rolling PR-AUC",
    )

    axis1.set_xlabel("Observation")
    axis1.set_ylabel("PR-AUC")

    axis2 = axis1.twinx()

    axis2.plot(
        centers,
        log_loss,
        linestyle="--",
        linewidth=2,
        label="Rolling Log Loss",
    )

    axis2.set_ylabel("Log Loss")

    axis1.set_title(
        f"Online Learning Performance "
        f"(rolling window = {window:,})"
    )

    axis1.grid(alpha=0.25)

    # Combine legends from both axes.
    handles1, labels1 = axis1.get_legend_handles_labels()
    handles2, labels2 = axis2.get_legend_handles_labels()

    axis1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="best",
    )

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostic plots from "
            "Online Logistic Regression predictions."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/evaluation/online_lr/predictions.jsonl"
        ),
        help="Path to predictions.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/online_lr/diagnostics"
        ),
        help="Directory for generated plots",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=2000,
        help="Window size for rolling metrics",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {args.input}"
        )

    if args.rolling_window <= 0:
        raise ValueError(
            "--rolling-window must be positive"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading predictions from: {args.input}")

    records = load_predictions(args.input)

    probabilities = np.asarray(
        [float(r["probability"]) for r in records],
        dtype=float,
    )

    y_true = np.asarray(
        [int(r["true_label"]) for r in records],
        dtype=int,
    )

    print(f"Observations: {len(records):,}")
    print(f"Positive:     {np.sum(y_true):,}")
    print(f"Negative:     {np.sum(y_true == 0):,}")

    print(
        f"Probability min:  {probabilities.min():.6f}"
    )
    print(
        f"Probability max:  {probabilities.max():.6f}"
    )
    print(
        f"Probability mean: {probabilities.mean():.6f}"
    )
    print(
        f"Probability median: "
        f"{np.median(probabilities):.6f}"
    )

    plot_probability_distribution(
        probabilities,
        y_true,
        args.output / "probability_distribution.png",
    )

    plot_precision_recall(
        y_true,
        probabilities,
        args.output / "precision_recall_curve.png",
    )

    plot_threshold_metrics(
        y_true,
        probabilities,
        args.output / "threshold_metrics.png",
    )

    plot_rolling_performance(
        y_true,
        probabilities,
        args.output / "rolling_performance.png",
        args.rolling_window,
    )

    print("\nGenerated:")
    for path in sorted(args.output.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()