
from pathlib import Path

import numpy as np
import pandas as pd


def generate_change_size_data():
    rng = np.random.default_rng(42)

    total_points = 120
    change_index = 60

    timestamps = pd.date_range(
        start="2026-01-03 09:00:00",
        periods=total_points,
        freq="5min",
    )

    baseline = rng.normal(180, 10, total_points)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for change_size in [30, 70, 150]:
        response_time = baseline.copy()
        response_time[change_index:] += change_size

        df = pd.DataFrame({
            "timestamp": timestamps,
            "response_time_ms": response_time.round(2),
            "is_after_change": [
                1 if i >= change_index else 0
                for i in range(total_points)
            ],
        })

        output_path = output_dir / f"change_size_{change_size}ms.csv"
        df.to_csv(output_path, index=False)

        print(
            f"Created {output_path.name}: "
            f"+{change_size} ms, "
            f"{len(df)} observations"
        )


if __name__ == "__main__":
    generate_change_size_data()
