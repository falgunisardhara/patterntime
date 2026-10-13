
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data():
    rng = np.random.default_rng(42)

    total_points = 120
    change_index = 72

    timestamps = pd.date_range(
        start="2026-01-01 09:00:00",
        periods=total_points,
        freq="5min",
    )

    # Normal application response time, in milliseconds.
    response_time = 180 + rng.normal(0, 10, total_points)

    # Simulate a slowdown beginning at a known point.
    response_time[change_index:] += 250

    # Simulate failed requests and total request volume.
    failed_requests = rng.poisson(2, total_points)
    failed_requests[change_index:] += rng.poisson(
        5, total_points - change_index
    )

    requests_per_minute = rng.normal(
        100, 8, total_points
    )
    requests_per_minute = np.maximum(
        requests_per_minute, 0
    ).round().astype(int)

    # Ground-truth label for evaluating detection later.
    is_after_change = np.zeros(total_points, dtype=int)
    is_after_change[change_index:] = 1

    df = pd.DataFrame({
        "timestamp": timestamps,
        "response_time_ms": response_time.round(2),
        "failed_requests": failed_requests,
        "requests_per_minute": requests_per_minute,
        "is_after_change": is_after_change,
    })

    output_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sample_api_metrics.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset created: {output_path}")
    print(f"Total observations: {len(df)}")
    print(f"Known change begins at: {timestamps[change_index]}")
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    generate_sample_data()
