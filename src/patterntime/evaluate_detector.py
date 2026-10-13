
from pathlib import Path

import pandas as pd

from detect_changes import detect_changes, group_detections


THRESHOLDS = [50, 75, 100]


def evaluate_dataset(data_path: Path, dataset_name: str, threshold: float):
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    metric = "response_time_ms"

    detections = detect_changes(
        df[metric],
        window_size=12,
        threshold=threshold,
    )

    events = group_detections(detections, max_gap=1)
    has_known_change = (df["is_after_change"] == 1).any()

    print(f"\n{dataset_name} | Threshold: {threshold} ms")
    print("-" * 45)
    print(f"Candidate detections: {len(detections)}")
    print(f"Grouped events: {len(events)}")

    if has_known_change:
        true_index = int(df.index[df["is_after_change"] == 1][0])

        if events:
            strongest_event = max(
                events,
                key=lambda event: abs(event["difference"]),
            )
            detected_index = strongest_event["index"]
            error_minutes = (detected_index - true_index) * 5

            print(f"Known change index: {true_index}")
            print(f"Detected index: {detected_index}")
            print(f"Detection error: {error_minutes} minutes")
        else:
            print("Result: Missed the known change.")
    else:
        print(f"False alarms: {len(events)}")


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
