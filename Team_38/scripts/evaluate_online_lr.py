"""Run online logistic-regression evaluation on PSV patient streams."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from clinical_data_gen.anomaly.anomaly_agent import AnomalyAgent
from clinical_data_gen.evaluation.runner import OnlineEvaluationRunner
from clinical_data_gen.evaluation.stream import (
    PSVEvaluationStream,
    discover_psv_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the online logistic-regression anomaly "
            "agent on PSV patient streams."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Directory containing patient PSV files, "
            "or a single PSV file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where evaluation artifacts are written.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Initial online learning rate.",
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="L2 regularization strength.",
    )

    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=1.0,
        help="Weight applied to positive examples.",
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
        help="Maximum absolute gradient component.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary predictions.",
    )

    parser.add_argument(
        "--constant-learning-rate",
        action="store_true",
        help=(
            "Disable diminishing learning rate and use "
            "the configured learning rate for every update."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help=(
            "Update the terminal progress display every N patients."
        ),
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patient files to evaluate. "
            "Useful for fast development runs; default is all patients.",
    )

    return parser.parse_args()


def build_agent(args: argparse.Namespace) -> AnomalyAgent:
    """Create the online anomaly agent."""

    return AnomalyAgent(
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_class_weight=args.positive_class_weight,
        gradient_clip=args.gradient_clip,
        probability_threshold=args.threshold,
        diminishing_learning_rate=(
            not args.constant_learning_rate
        ),
    )


def write_run_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    patient_files: list[Path],
    agent: AnomalyAgent,
) -> None:
    """Write reproducibility metadata for the evaluation run."""

    metadata = {
        "input": str(args.input),
        "patient_file_count": len(patient_files),
        "model": "online_logistic_regression",
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "positive_class_weight": args.positive_class_weight,
        "gradient_clip": args.gradient_clip,
        "probability_threshold": args.threshold,
        "diminishing_learning_rate": (
            not args.constant_learning_rate
        ),
        "feature_count": agent.feature_extractor.n_features,
        "feature_names": list(
            agent.feature_extractor.feature_names
        ),
        "evaluation_protocol": "prequential",
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""

    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_bar(
    current: int,
    total: int,
    width: int = 35,
) -> str:
    """Build a terminal progress bar."""

    if total <= 0:
        return "[" + ("?" * width) + "]"

    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)

    return (
        "["
        + ("#" * filled)
        + ("-" * (width - filled))
        + "]"
    )


def _print_progress(
    *,
    patient_index: int,
    total_patients: int,
    observation_count: int,
    start_time: float,
    runner: OnlineEvaluationRunner,
) -> None:
    """Print the current evaluation/training state."""

    elapsed = time.perf_counter() - start_time

    if patient_index > 0:
        rate = patient_index / elapsed if elapsed > 0 else 0.0

        remaining_patients = (
            total_patients - patient_index
        )

        eta = (
            remaining_patients / rate
            if rate > 0
            else 0.0
        )
    else:
        rate = 0.0
        eta = 0.0

    evaluator = runner.evaluator

    if evaluator.records:
        metrics = evaluator.compute()

        precision = metrics.precision
        recall = metrics.recall
        f1 = metrics.f1
        log_loss = metrics.mean_loss
        pr_auc = metrics.pr_auc
    else:
        precision = recall = f1 = log_loss = 0.0
        pr_auc = None

    percentage = (
        100.0 * patient_index / total_patients
        if total_patients
        else 0.0
    )

    pr_auc_text = (
        f"{pr_auc:.4f}"
        if pr_auc is not None
        else "N/A"
    )

    bar = _progress_bar(
        patient_index,
        total_patients,
    )

    message = (
        f"\r"
        f"{bar} "
        f"{percentage:6.2f}% "
        f"{patient_index:,}/{total_patients:,} patients"
        f" | observations={observation_count:,}"
        f" | PR-AUC={pr_auc_text}"
        f" | P={precision:.3f}"
        f" | R={recall:.3f}"
        f" | F1={f1:.3f}"
        f" | loss={log_loss:.4f}"
        f" | {rate:.2f} patients/s"
        f" | ETA={_format_duration(eta)}"
    )

    sys.stdout.write(message)
    sys.stdout.flush()


def _evaluate_with_progress(
    *,
    patient_files: list[Path],
    runner: OnlineEvaluationRunner,
    progress_every: int,
) -> None:
    """
    Evaluate patients while displaying online-learning progress.

    The model is updated after every observation. The progress display
    is updated after every N patients.
    """

    start_time = time.perf_counter()
    observation_count = 0

    total_patients = len(patient_files)

    print()
    print("Starting online evaluation...")
    print(
        "The model predicts first, then learns from the label "
        "for every observation."
    )
    print()

    for patient_index, psv_path in enumerate(
        patient_files,
        start=1,
    ):
        stream = PSVEvaluationStream(psv_path)

        before = len(runner.evaluator.records)

        runner.evaluate(
            stream.samples()
        )

        after = len(runner.evaluator.records)

        observation_count += after - before

        should_update = (
            patient_index % progress_every == 0
            or patient_index == total_patients
        )

        if should_update:
            _print_progress(
                patient_index=patient_index,
                total_patients=total_patients,
                observation_count=observation_count,
                start_time=start_time,
                runner=runner,
            )

    elapsed = time.perf_counter() - start_time

    print()
    print()
    print(
        f"Evaluation completed in "
        f"{_format_duration(elapsed)}."
    )


def main() -> None:
    args = parse_args()

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be greater than zero"
        )

    patient_files = discover_psv_files(args.input)

    if args.max_patients is not None:
        patient_files = patient_files[:args.max_patients]

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent = build_agent(args)

    write_run_metadata(
        args.output,
        args=args,
        patient_files=patient_files,
        agent=agent,
    )

    runner = OnlineEvaluationRunner(agent)

    print("=" * 80)
    print("ONLINE LOGISTIC REGRESSION EVALUATION")
    print("=" * 80)
    print(f"Input          : {args.input}")
    print(f"Patients       : {len(patient_files):,}")
    print(f"Output         : {args.output}")
    print(f"Features       : {agent.feature_extractor.n_features}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Positive weight: {args.positive_class_weight}")
    print(f"Threshold      : {args.threshold}")
    print()

    _evaluate_with_progress(
        patient_files=patient_files,
        runner=runner,
        progress_every=args.progress_every,
    )

    artifacts = runner.save_results(
        args.output,
    )

    metrics = runner.evaluator.compute()

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"Observations   : {metrics.sample_count:,}")
    print(f"Positive       : {metrics.positive_count:,}")
    print(f"Negative       : {metrics.negative_count:,}")
    print()

    print(f"Accuracy       : {metrics.accuracy:.4f}")
    print(
        f"Balanced Acc.  : "
        f"{metrics.balanced_accuracy:.4f}"
    )
    print(f"Precision      : {metrics.precision:.4f}")
    print(f"Recall         : {metrics.recall:.4f}")
    print(f"Specificity    : {metrics.specificity:.4f}")
    print(f"F1             : {metrics.f1:.4f}")

    if metrics.roc_auc is not None:
        print(f"ROC-AUC        : {metrics.roc_auc:.4f}")
    else:
        print("ROC-AUC        : N/A")

    if metrics.pr_auc is not None:
        print(f"PR-AUC         : {metrics.pr_auc:.4f}")
    else:
        print("PR-AUC         : N/A")

    print(f"Brier Score    : {metrics.brier_score:.4f}")
    print(f"Log Loss       : {metrics.log_loss:.4f}")
    print(
        "Calibration ECE: "
        f"{metrics.expected_calibration_error:.4f}"
    )
    print(f"Mean Loss      : {metrics.mean_loss:.4f}")

    if metrics.alerts_per_patient_hour is not None:
        print(
            "Alerts/hour    : "
            f"{metrics.alerts_per_patient_hour:.4f}"
        )
    else:
        print("Alerts/hour    : N/A")

    print()
    print("Confusion Matrix")
    print(
        f"  TN={metrics.confusion_matrix.true_negative:,} "
        f"FP={metrics.confusion_matrix.false_positive:,}"
    )
    print(
        f"  FN={metrics.confusion_matrix.false_negative:,} "
        f"TP={metrics.confusion_matrix.true_positive:,}"
    )

    print()
    print("Artifacts")
    for name, path in artifacts.items():
        print(f"  {name:18s}: {path}")

    print()


if __name__ == "__main__":
    main()