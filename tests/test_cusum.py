
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "patterntime"))

from detect_cusum import detect_cusum


DATA_DIR = PROJECT_ROOT / "data"


def load_dataset(filename):
    return pd.read_csv(DATA_DIR / filename)


def test_no_change_dataset_has_no_false_alarm_at_7_5():
    df = load_dataset("no_change_metrics.csv")

    detections = detect_cusum(
        df["response_time_ms"],
        baseline_window=20,
        drift=0.5,
        threshold=7.5,
    )

    assert detections == []


@pytest.mark.parametrize(
    ("filename", "expected_index"),
    [
        ("change_size_30ms.csv", 63),
        ("change_size_70ms.csv", 61),
        ("change_size_150ms.csv", 60),
    ],
)
def test_cusum_detects_injected_upward_changes(filename, expected_index):
    df = load_dataset(filename)

    detections = detect_cusum(
        df["response_time_ms"],
        baseline_window=20,
        drift=0.5,
        threshold=7.5,
    )

    assert detections
    assert detections[0]["direction"] == "up"
    assert detections[0]["index"] == expected_index


def test_cusum_rejects_invalid_threshold():
    values = pd.Series([10, 11, 9, 10, 11, 12])

    with pytest.raises(ValueError, match="threshold"):
        detect_cusum(
            values,
            baseline_window=2,
            threshold=0,
        )


def test_cusum_rejects_missing_values():
    values = pd.Series([10.0, 11.0, None, 12.0, 11.0])

    with pytest.raises(ValueError, match="missing values"):
        detect_cusum(values, baseline_window=2)


def test_cusum_rejects_invalid_clip_z_score():
    values = pd.Series([10, 11, 9, 10, 11, 12])

    with pytest.raises(ValueError, match="clip_z_score"):
        detect_cusum(values, baseline_window=2, clip_z_score=0)

    with pytest.raises(ValueError, match="clip_z_score"):
        detect_cusum(values, baseline_window=2, clip_z_score=-2.5)


def test_robust_cusum_suppresses_transient_spikes():
    df = load_dataset("scenario_transient_spikes.csv")

    # Standard CUSUM without clipping is fooled by isolated spikes
    standard_detections = detect_cusum(
        df["response_time_ms"],
        baseline_window=20,
        drift=0.5,
        threshold=10.0,
        clip_z_score=None,
    )
    assert len(standard_detections) == 2

    # Robust CUSUM with Z-clamping ignores isolated spikes
    robust_detections = detect_cusum(
        df["response_time_ms"],
        baseline_window=20,
        drift=0.5,
        threshold=10.0,
        clip_z_score=4.0,
    )
    assert robust_detections == []

