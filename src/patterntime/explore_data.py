
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "sample_api_metrics.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run generate_sample_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    required_columns = [
        "timestamp",
        "response_time_ms",
        "failed_requests",
        "requests_per_minute",
        "is_after_change",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("=== Dataset Overview ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Time range: {df['timestamp'].min()} to "
          f"{df['timestamp'].max()}")
    print(f"Duplicate timestamps: {df['timestamp'].duplicated().sum()}")

    print("\n=== Missing Values ===")
    print(df[required_columns].isna().sum())

    numeric_metrics = [
        "response_time_ms",
        "failed_requests",
        "requests_per_minute",
    ]

    print("\n=== Metric Summary ===")
    print(df[numeric_metrics].describe().round(2))

    before = df[df["is_after_change"] == 0]
    after = df[df["is_after_change"] == 1]

    if before.empty or after.empty:
        raise ValueError(
            "Both before-change and after-change observations are required."
        )

    change_time = after["timestamp"].min()

    print("\n=== Known Change ===")
    print(f"Change begins at: {change_time}")
    print(f"Observations before: {len(before)}")
    print(f"Observations after: {len(after)}")

    print("\n=== Before vs After ===")

    for metric in numeric_metrics:
        before_mean = before[metric].mean()
        after_mean = after[metric].mean()
        difference = after_mean - before_mean

        print(f"\n{metric}")
        print(f"  Before mean: {before_mean:.2f}")
        print(f"  After mean:  {after_mean:.2f}")
        print(f"  Difference:  {difference:+.2f}")


if __name__ == "__main__":
    main()
