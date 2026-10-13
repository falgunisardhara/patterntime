
from pathlib import Path

import numpy as np
import pandas as pd


def generate_no_change_data():
    rng = np.random.default_rng(123)

    total_points = 120

    timestamps = pd.date_range(
        start="2026-01-02 09:00:00",
        periods=total_points,
        freq="5min",
    )

    # Stable average response time with ordinary random fluctuations.
    response_time = rng.normal(180, 10, total_points)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "response_time_ms": response_time.round(2),
        "is_after_change": 0,
    })

    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "no_change_metrics.csv"

    df.to_csv(output_path, index=False)

    print(f"Dataset created: {output_path}")
    print(f"Observations: {len(df)}")
    print(f"Average response time: {df['response_time_ms'].mean():.2f} ms")
    print("Known changes: 0")


if __name__ == "__main__":
    generate_no_change_data()
