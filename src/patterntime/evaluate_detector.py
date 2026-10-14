
from pathlib import Path

import pandas as pd

from detect_changes import detect_changes, group_detections


THRESHOLDS = [50, 75, 100]
WINDOW_SIZE = 12
MATCH_TOLERANCE = 2  # observations; each observation is 5 minutes


def get_known_change_indices(df: pd.DataFrame) -> list[int]:
    """Find indices where the ground-truth label changes."""

    labels = df["is_after_change"].astype(int)

    return [
        index
        for index in range(1, len(labels))
        if labels.iloc[index] != labels.iloc[index - 1]
    ]


def match_events(
    detected_indices: list[int],
    true_indices: list[int],
    tolerance: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detected events to true changes in chronological order."""

    detected = sorted(detected_indices)
    actual = sorted(true_indices)

    matches = []
    false_positives = []
    false_negatives = []

    i = 0
    j = 0

    while i < len(detected) and j < len(actual):
        difference = detected[i] - actual[j]

        if abs(difference) <= tolerance:
            matches.append((actual[j], detected[i]))
            i += 1
            j += 1
        elif difference < -tolerance:
            false_positives.append(detected[i])
            i += 1
        else:
            false_negatives.append(actual[j])
            j += 1

    false_positives.extend(detected[i:])
    false_negatives.extend(actual[j:])

    return matches, false_positives, false_negatives


def calculate_metrics(
    matches: list[tuple[int, int]],
    false_positives: list[int],
    false_negatives: list[int],
    true_change_count: int,
) -> dict:
    """Calculate event-level detection metrics."""

    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / true_change_count if true_change_count > 0 else None

    f1_denominator = (2 * tp) + fp + fn
    f1 = (2 * tp / f1_denominator) if f1_denominator > 0 else None

    errors = [
        detected - actual
        for actual, detected in matches
    ]

    mean_absolute_error = (
        sum(abs(error) for error in errors) / len(errors)
        if errors
        else None
    )

    mean_signed_error = (
        sum(errors) / len(errors)
        if errors
        else None
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_error_minutes": (
            mean_absolute_error * 5
            if mean_absolute_error is not None
            else None
        ),
        "mean_signed_error_minutes": (
            mean_signed_error * 5
            if mean_signed_error is not None
            else None
        ),
    }


def format_metric(value) -> str:
    """Format a metric or show N/A when it is undefined."""

    return f"{value:.2f}" if value is not None else "N/A"


def evaluate_dataset(
    data_path: Path,
    dataset_name: str,
    threshold: float,
) -> dict:
    """Run the detector and evaluate its grouped events."""

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    metric = "response_time_ms"

    detections = detect_changes(
        df[metric],
        window_size=WINDOW_SIZE,
        threshold=threshold,
    )

    events = group_detections(detections, max_gap=1)

    true_indices = get_known_change_indices(df)
    detected_indices = [event["index"] for event in events]

    matches, false_positives, false_negatives = match_events(
        detected_indices,
        true_indices,
        MATCH_TOLERANCE,
    )

    metrics = calculate_metrics(
        matches,
        false_positives,
        false_negatives,
        len(true_indices),
    )

    print(f"\n{dataset_name}")
    print("-" * 45)
    print(f"Threshold: {threshold} ms")
    print(f"Observations: {len(df)}")
    print(f"Known changes: {len(true_indices)}")
    print(f"Candidate detections: {len(detections)}")
    print(f"Grouped events: {len(events)}")
    print(f"Matching tolerance: {MATCH_TOLERANCE * 5} minutes")
    print(f"True positives (TP): {metrics['true_positives']}")
    print(f"False positives (FP): {metrics['false_positives']}")
    print(f"False negatives (FN): {metrics['false_negatives']}")
    print(f"Precision: {format_metric(metrics['precision'])}")
    print(f"Recall: {format_metric(metrics['recall'])}")
    print(f"F1 score: {format_metric(metrics['f1'])}")

    if metrics["mean_absolute_error_minutes"] is not None:
        print(
            "Mean absolute timing error: "
            f"{metrics['mean_absolute_error_minutes']:.2f} minutes"
        )
        print(
            "Mean signed timing error: "
            f"{metrics['mean_signed_error_minutes']:.2f} minutes"
        )
    else:
        print("Timing error: N/A (no matched events)")

    return metrics


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    datasets = [
        ("Original Dataset", "sample_api_metrics.csv"),
        ("No-Change Dataset", "no_change_metrics.csv"),
        ("Small Change (+30 ms)", "change_size_30ms.csv"),
        ("Medium Change (+70 ms)", "change_size_70ms.csv"),
        ("Large Change (+150 ms)", "change_size_150ms.csv"),
    ]

    for threshold in THRESHOLDS:
        print("\n" + "=" * 55)
        print(f"EVALUATING THRESHOLD: {threshold} ms")
        print("=" * 55)

        for dataset_name, filename in datasets:
            evaluate_dataset(
                data_dir / filename,
                dataset_name,
                threshold,
            )


if __name__ == "__main__":
    main()
