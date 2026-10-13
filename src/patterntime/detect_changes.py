
from pathlib import Path

import pandas as pd


def detect_changes(
    values: pd.Series,
    window_size: int = 12,
    threshold: float = 100.0,
) -> list[dict]:
    """Find candidate changes by comparing adjacent windows."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    if threshold < 0:
        raise ValueError("threshold cannot be negative.")

    if values.isna().any():
        raise ValueError("Input values must not contain missing values.")

    if len(values) < window_size * 2:
        raise ValueError("Not enough observations for two windows.")

    detections = []

    for index in range(window_size, len(values) - window_size + 1):
        earlier = values.iloc[index - window_size:index]
        recent = values.iloc[index:index + window_size]

        earlier_mean = float(earlier.mean())
        recent_mean = float(recent.mean())
        difference = recent_mean - earlier_mean

        if abs(difference) >= threshold:
            detections.append({
                "index": index,
                "earlier_mean": round(earlier_mean, 2),
                "recent_mean": round(recent_mean, 2),
                "difference": round(difference, 2),
            })

    return detections


def group_detections(
    detections: list[dict],
    max_gap: int = 1,
) -> list[dict]:
    """Group consecutive candidates and retain the strongest one."""

    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item["index"])
    groups = []
    current_group = [ordered[0]]

    for detection in ordered[1:]:
        previous_index = current_group[-1]["index"]

        if detection["index"] - previous_index <= max_gap:
            current_group.append(detection)
        else:
            groups.append(current_group)
            current_group = [detection]

    groups.append(current_group)

    events = []

    for group in groups:
        strongest = max(
            group,
            key=lambda item: abs(item["difference"]),
        )
        events.append(strongest)

    return events


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "sample_api_metrics.csv"

    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    metric = "response_time_ms"

    detections = detect_changes(
        df[metric],
        window_size=12,
        threshold=100,
    )

    events = group_detections(detections, max_gap=1)

    print(f"Metric: {metric}")
    print(f"Observations analyzed: {len(df)}")
    print("Window size: 12 observations")
    print("Detection threshold: 100 ms")
    print(f"Potential candidates found: {len(detections)}")
    print(f"Grouped change events: {len(events)}")

    for event in events:
        timestamp = df.iloc[event["index"]]["timestamp"]

        print(
            f"Change candidate: {timestamp} | "
            f"Index: {event['index']} | "
            f"Difference: {event['difference']} ms"
        )


if __name__ == "__main__":
    main()
