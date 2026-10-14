
from pathlib import Path

import pandas as pd

from detect_changes import detect_changes, group_detections
from detect_cusum import detect_cusum


WINDOW_SIZE = 12
SLIDING_THRESHOLD = 50.0

CUSUM_BASELINE_WINDOW = 20
CUSUM_DRIFT = 0.5
CUSUM_THRESHOLD = 10.0

MATCH_TOLERANCE = 2
MINUTES_PER_OBSERVATION = 5


def get_known_change_indices(df: pd.DataFrame) -> list[int]:
    """Return indices where the ground-truth label changes."""
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

    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator > 0 else None

    errors = [detected - actual for actual, detected in matches]

    mean_absolute_error = (
        sum(abs(error) for error in errors) / len(errors)
        if errors else None
    )
    mean_signed_error = (
        sum(errors) / len(errors)
        if errors else None
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_error_minutes": (
            mean_absolute_error * MINUTES_PER_OBSERVATION
            if mean_absolute_error is not None else None
        ),
        "mean_signed_error_minutes": (
            mean_signed_error * MINUTES_PER_OBSERVATION
            if mean_signed_error is not None else None
        ),
    }


def format_metric(value) -> str:
    """Format a metric, displaying N/A when undefined."""
    return f"{value:.2f}" if value is not None else "N/A"


def evaluate_algorithm(
    df: pd.DataFrame,
    algorithm: str,
    true_indices: list[int],
) -> dict:
    """Run one detector and evaluate its grouped or reported events."""
    values = df["response_time_ms"]

    if algorithm == "Sliding Window":
        candidates = detect_changes(
            values,
            window_size=WINDOW_SIZE,
            threshold=SLIDING_THRESHOLD,
        )
        events = group_detections(candidates, max_gap=1)
        detected_indices = [event["index"] for event in events]
        candidate_count = len(candidates)

    elif algorithm == "CUSUM":
        events = detect_cusum(
            values,
            baseline_window=CUSUM_BASELINE_WINDOW,
            drift=CUSUM_DRIFT,
            threshold=CUSUM_THRESHOLD,
        )
        detected_indices = [event["index"] for event in events]
        candidate_count = len(events)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

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

    return {
        "algorithm": algorithm,
        "candidate_count": candidate_count,
        "detected_indices": detected_indices,
        **metrics,
    }


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

    all_results = []

    print("=" * 100)
    print("PATTERNTIME: SLIDING-WINDOW VS CUSUM")
    print("=" * 100)
    print(f"Sliding window: size={WINDOW_SIZE}, threshold={SLIDING_THRESHOLD} ms")
    print(
        f"CUSUM: baseline={CUSUM_BASELINE_WINDOW}, "
        f"drift={CUSUM_DRIFT}, threshold={CUSUM_THRESHOLD}"
    )
    print(
        f"Event matching tolerance: {MATCH_TOLERANCE} observations "
        f"({MATCH_TOLERANCE * MINUTES_PER_OBSERVATION} minutes)"
    )

    for dataset_name, filename in datasets:
        df = pd.read_csv(data_dir / filename, parse_dates=["timestamp"])
        true_indices = get_known_change_indices(df)

        print(f"\n{'-' * 100}")
        print(f"{dataset_name} | Known change indices: {true_indices}")
        print("-" * 100)

        for algorithm in ["Sliding Window", "CUSUM"]:
            result = evaluate_algorithm(df, algorithm, true_indices)
            result["dataset"] = dataset_name
            all_results.append(result)

            print(f"\n{algorithm}")
            print(f"  Detected indices: {result['detected_indices']}")
            print(f"  Candidate detections: {result['candidate_count']}")
            print(
                f"  TP: {result['true_positives']} | "
                f"FP: {result['false_positives']} | "
                f"FN: {result['false_negatives']}"
            )
            print(
                f"  Precision: {format_metric(result['precision'])} | "
                f"Recall: {format_metric(result['recall'])} | "
                f"F1: {format_metric(result['f1'])}"
            )
            print(
                "  Mean absolute localization error: "
                f"{format_metric(result['mean_absolute_error_minutes'])} minutes"
            )
            print(
                "  Mean signed localization error: "
                f"{format_metric(result['mean_signed_error_minutes'])} minutes"
            )

    print(f"\n{'=' * 100}")
    print("COMPARISON SUMMARY")
    print("=" * 100)

    summary = pd.DataFrame(all_results)
    columns = [
        "dataset",
        "algorithm",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_absolute_error_minutes",
    ]

    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
