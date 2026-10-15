
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "patterntime"))

from detect_cusum import detect_cusum


data_dir = PROJECT_ROOT / "data"

datasets = [
    ("Original Dataset", "sample_api_metrics.csv"),
    ("No change", "no_change_metrics.csv"),
    ("Small change (+30 ms)", "change_size_30ms.csv"),
    ("Medium change (+70 ms)", "change_size_70ms.csv"),
    ("Large change (+150 ms)", "change_size_150ms.csv"),
]

for threshold in [5.0, 7.5, 10.0]:
    print("\n" + "=" * 55)
    print(f"CUSUM THRESHOLD: {threshold}")
    print("=" * 55)

    for name, filename in datasets:
        df = pd.read_csv(data_dir / filename)
        detections = detect_cusum(
            df["response_time_ms"],
            baseline_window=20,
            drift=0.5,
            threshold=threshold,
        )

        indices = [event["index"] for event in detections]
        directions = [event["direction"] for event in detections]

        print(f"\n{name}")
        print(f"Known change indices: {df.index[df['is_after_change'].diff().fillna(0) != 0].tolist()}")
        print(f"Detected indices: {indices}")
        print(f"Directions: {directions}")
