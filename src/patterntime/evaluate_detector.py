
from pathlib import Path

import pandas as pd

from detect_changes import detect_changes, group_detections


def evaluate_dataset(data_path: Path, dataset_name: str):
    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    metric = "response_time_ms"

    detections = detect_changes(
        df[metric],
        window_size=12,
        threshold=100,
    )
    events = group_detections(detections, max_gap=1)

    has_known_change = (df["is_after_change"] == 1).any()

    print(f"\n{dataset_name}")
    print("-" * 40)
    print(f"Observations: {len(df)}")
    print(f"Known change exists: {has_known_change}")
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
            error = detected_index - true_index

            print(f"Known change index: {true_index}")
            print(f"Detected index: {detected_index}")
            print(f"Detection error: {error * 5} minutes")
        else:
            print("The detector missed the known change.")
    else:
        print(f"False alarms: {len(events)}")

        if events:
            print("The detector reported a change despite no known change.")
        else:
            print("No false alarms in this dataset.")



def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    evaluate_dataset(
        data_dir / "sample_api_metrics.csv",
        "Original Dataset: Injected Change",
    )

    evaluate_dataset(
        data_dir / "no_change_metrics.csv",
        "No-Change Dataset",
    )

    for change_size in [30, 70, 150]:
        evaluate_dataset(
            data_dir / f"change_size_{change_size}ms.csv",
            f"Change-Size Experiment: +{change_size} ms",
        )


if __name__ == "__main__":
    main()

