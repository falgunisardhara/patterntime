
from pathlib import Path

import pandas as pd

from detect_changes import detect_changes, group_detections


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "sample_api_metrics.csv"

    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    metric = "response_time_ms"
    window_size = 12
    threshold = 100

    # Run the detector and group nearby candidates.
    detections = detect_changes(
        df[metric],
        window_size=window_size,
        threshold=threshold,
    )
    events = group_detections(detections, max_gap=1)

    # The synthetic dataset tells us where the change was introduced.
    true_change_index = int(
        df.index[df["is_after_change"] == 1][0]
    )

    print("PATTERNTIME — DETECTOR EVALUATION")
    print("-" * 40)
    print(f"Metric: {metric}")
    print(f"Known change index: {true_change_index}")
    print(f"Known change time: {df.iloc[true_change_index]['timestamp']}")
    print(f"Candidate detections: {len(detections)}")
    print(f"Grouped events: {len(events)}")

    if not events:
        print("Result: No change detected.")
        return

    # Select the strongest grouped event for this first experiment.
    strongest_event = max(
        events,
        key=lambda event: abs(event["difference"]),
    )

    detected_index = strongest_event["index"]
    error_points = detected_index - true_change_index
    error_minutes = error_points * 5

    print(f"Detected index: {detected_index}")
    print(f"Detected time: {df.iloc[detected_index]['timestamp']}")
    print(f"Detection error: {error_points} observations")
    print(f"Detection error in time: {error_minutes} minutes")

    if error_points == 0:
        print("Result: Exact match with the known change.")
    else:
        print("Result: Detection differs from the known change.")


if __name__ == "__main__":
    main()
