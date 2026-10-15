from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patterntime.detectors import (
    BaseDetector,
    ChangeEvent,
    CusumDetector,
    DetectionResult,
    EWMADetector,
    RobustCusumDetector,
    SlidingWindowDetector,
    get_detector,
    list_available_detectors,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Test Data Models & Structure
# ---------------------------------------------------------------------------
def test_change_event_to_dict():
    ts = pd.Timestamp("2026-01-01 12:00:00")
    event = ChangeEvent(
        index=42,
        timestamp=ts,
        direction="up",
        score=7.891234,
        metadata={"diff": 12.34567, "algo": "test"},
    )
    d = event.to_dict()

    assert d["index"] == 42
    assert d["timestamp"] == "2026-01-01T12:00:00"
    assert d["direction"] == "up"
    assert d["score"] == 7.8912
    assert d["metadata"]["diff"] == 12.3457
    assert d["metadata"]["algo"] == "test"


def test_detection_result_properties_and_serialization():
    events = [
        ChangeEvent(index=10, direction="up", score=5.0),
        ChangeEvent(index=30, direction="down", score=6.2),
    ]
    result = DetectionResult(
        events=events,
        algorithm_name="Mock Algo",
        params={"threshold": 5.0},
        execution_time_ms=1.234,
    )

    assert result.indices == [10, 30]
    assert result.count == 2
    assert result.has_changes is True
    assert result.timestamps == []

    d = result.to_dict()
    assert d["algorithm_name"] == "Mock Algo"
    assert d["total_changes"] == 2
    assert d["indices"] == [10, 30]
    assert d["execution_time_ms"] == 1.234
    assert len(d["events"]) == 2


# ---------------------------------------------------------------------------
# Test Base Validation & Timestamp Attachment
# ---------------------------------------------------------------------------
def test_empty_series_raises_error():
    detector = SlidingWindowDetector()
    with pytest.raises(ValueError, match="cannot be empty"):
        detector.detect(pd.Series([]))


def test_missing_values_raise_error():
    detector = CusumDetector()
    series = pd.Series([10.0, 11.0, np.nan, 12.0])
    with pytest.raises(ValueError, match="must not contain NaN"):
        detector.detect(series)


def test_mismatched_timestamps_length_raises_error():
    detector = CusumDetector()
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    ts = pd.Series(pd.date_range("2026-01-01", periods=3, freq="1h"))
    with pytest.raises(ValueError, match="length.*does not match"):
        detector.detect(series, timestamps=ts)


def test_timestamps_attached_to_events():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")
    detector = SlidingWindowDetector(window_size=12, threshold=50.0)

    result = detector.detect(df["response_time_ms"], timestamps=df["timestamp"])

    assert result.has_changes is True
    assert len(result.timestamps) == len(result.events)
    for event in result.events:
        assert isinstance(event.timestamp, pd.Timestamp)
        expected_ts = pd.to_datetime(df["timestamp"].iloc[event.index])
        assert event.timestamp == expected_ts


# ---------------------------------------------------------------------------
# Test Individual Detectors
# ---------------------------------------------------------------------------
def test_sliding_window_detects_shift():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")
    detector = SlidingWindowDetector(window_size=12, threshold=80.0)

    result = detector.detect(df["response_time_ms"])

    assert result.algorithm_name == "Sliding Window"
    assert result.count >= 1
    # Ground truth change in sample_api_metrics is at index 72
    assert 70 <= result.indices[0] <= 74
    assert result.events[0].direction == "up"


def test_sliding_window_rejects_invalid_params():
    with pytest.raises(ValueError, match="window_size"):
        SlidingWindowDetector(window_size=0)
    with pytest.raises(ValueError, match="threshold"):
        SlidingWindowDetector(threshold=-1.0)
    with pytest.raises(ValueError, match="group_max_gap"):
        SlidingWindowDetector(group_max_gap=-1)


def test_cusum_detects_shift():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")
    detector = CusumDetector(baseline_window=20, drift=0.5, threshold=10.0)

    result = detector.detect(df["response_time_ms"])

    assert result.algorithm_name == "CUSUM"
    assert result.count == 1
    assert result.indices[0] == 72
    assert result.events[0].direction == "up"


def test_cusum_rejects_invalid_params():
    with pytest.raises(ValueError, match="baseline_window"):
        CusumDetector(baseline_window=1)
    with pytest.raises(ValueError, match="drift"):
        CusumDetector(drift=-0.1)
    with pytest.raises(ValueError, match="threshold"):
        CusumDetector(threshold=0)
    with pytest.raises(ValueError, match="clip_z_score"):
        CusumDetector(clip_z_score=-2.0)


def test_robust_cusum_suppresses_spikes():
    df = pd.read_csv(DATA_DIR / "scenario_transient_spikes.csv")

    # Standard CUSUM flags the isolated spikes
    standard_cusum = CusumDetector(baseline_window=20, drift=0.5, threshold=10.0)
    std_result = standard_cusum.detect(df["response_time_ms"])
    assert std_result.count == 2

    # Robust CUSUM with Winsorization ignores the spikes
    robust_cusum = RobustCusumDetector(
        baseline_window=20, drift=0.5, threshold=10.0, clip_z_score=4.0
    )
    robust_result = robust_cusum.detect(df["response_time_ms"])
    assert robust_result.count == 0
    assert robust_result.has_changes is False


def test_ewma_detects_gradual_drift():
    df = pd.read_csv(DATA_DIR / "scenario_gradual_drift.csv")

    detector = EWMADetector(
        baseline_window=20,
        lambda_=0.2,
        control_limit=3.0,
        clip_z_score=4.0,
    )

    result = detector.detect(df["response_time_ms"])

    assert result.algorithm_name == "EWMA"
    assert result.count >= 1
    # Gradual drift starts at index 40, EWMA detects it smoothly
    first_detected = result.indices[0]
    assert 40 <= first_detected <= 55
    assert result.events[0].direction == "up"


def test_ewma_no_false_alarms_on_clean_data():
    df = pd.read_csv(DATA_DIR / "no_change_metrics.csv")

    detector = EWMADetector(baseline_window=20, lambda_=0.2, control_limit=3.0)
    result = detector.detect(df["response_time_ms"])

    assert result.count == 0
    assert result.has_changes is False


def test_ewma_rejects_invalid_params():
    with pytest.raises(ValueError, match="baseline_window"):
        EWMADetector(baseline_window=1)
    with pytest.raises(ValueError, match="lambda_"):
        EWMADetector(lambda_=0.0)
    with pytest.raises(ValueError, match="lambda_"):
        EWMADetector(lambda_=1.5)
    with pytest.raises(ValueError, match="control_limit"):
        EWMADetector(control_limit=0)
    with pytest.raises(ValueError, match="clip_z_score"):
        EWMADetector(clip_z_score=-1.0)


# ---------------------------------------------------------------------------
# Test Factory Registry
# ---------------------------------------------------------------------------
def test_list_available_detectors():
    available = list_available_detectors()
    assert "sliding_window" in available
    assert "cusum" in available
    assert "robust_cusum" in available
    assert "ewma" in available


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        ("sliding_window", SlidingWindowDetector),
        ("SLIDING_WINDOW", SlidingWindowDetector),
        ("sliding-window", SlidingWindowDetector),
        ("cusum", CusumDetector),
        ("Cusum", CusumDetector),
        ("robust_cusum", RobustCusumDetector),
        ("ewma", EWMADetector),
        ("EWMA", EWMADetector),
    ],
)
def test_get_detector_factory(name: str, expected_cls: type[BaseDetector]):
    detector = get_detector(name)
    assert isinstance(detector, expected_cls)
    assert isinstance(detector, BaseDetector)


def test_get_detector_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown detector 'non_existent'"):
        get_detector("non_existent")
