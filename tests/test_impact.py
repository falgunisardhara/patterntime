from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patterntime.impact import (
    ChangeImpactReport,
    MetricImpact,
    SegmentStats,
    analyze_all_changes,
    analyze_change_impact,
    analyze_metric_shift,
    calculate_cohen_d,
    classify_severity,
    compute_segment_stats,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Test Segment Statistics
# ---------------------------------------------------------------------------
def test_compute_segment_stats_standard():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_segment_stats(series)

    assert stats.count == 5
    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert stats.min == 10.0
    assert stats.max == 50.0
    assert stats.p25 == 20.0
    assert stats.p75 == 40.0
    assert stats.iqr == 20.0
    assert np.isclose(stats.std, np.std(series, ddof=1))

    # Serialization test
    d = stats.to_dict()
    assert d["mean"] == 30.0
    assert d["count"] == 5


def test_compute_segment_stats_empty():
    stats = compute_segment_stats(pd.Series([]))
    assert stats.count == 0
    assert stats.mean == 0.0
    assert stats.std == 0.0


def test_compute_segment_stats_single_point():
    stats = compute_segment_stats(pd.Series([42.0]))
    assert stats.count == 1
    assert stats.mean == 42.0
    assert stats.std == 0.0


# ---------------------------------------------------------------------------
# Test Cohen's d & Effect Size
# ---------------------------------------------------------------------------
def test_cohen_d_identical_samples():
    s1 = pd.Series([10.0, 12.0, 11.0, 10.5])
    s2 = pd.Series([10.0, 12.0, 11.0, 10.5])
    d, category = calculate_cohen_d(s1, s2)

    assert np.isclose(d, 0.0)
    assert category == "negligible"


def test_cohen_d_large_shift():
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(100, 10, 50))
    s2 = pd.Series(rng.normal(150, 10, 50))  # Shift of 5 standard deviations
    d, category = calculate_cohen_d(s1, s2)

    assert d > 3.0
    assert category == "huge"


def test_cohen_d_zero_variance_different_means():
    s1 = pd.Series([10.0, 10.0, 10.0])
    s2 = pd.Series([20.0, 20.0, 20.0])
    d, category = calculate_cohen_d(s1, s2)

    assert d > 0.0
    assert category == "huge"


# ---------------------------------------------------------------------------
# Test Severity Classification
# ---------------------------------------------------------------------------
def test_classify_severity_levels():
    # Non-significant
    assert classify_severity(percent_change=50.0, cohen_d=2.0, p_value=0.15) == "negligible"

    # Significant shifts of varying magnitudes
    assert classify_severity(percent_change=2.0, cohen_d=0.1, p_value=0.01) == "negligible"
    assert classify_severity(percent_change=10.0, cohen_d=0.35, p_value=0.01) == "low"
    assert classify_severity(percent_change=25.0, cohen_d=0.7, p_value=0.001) == "medium"
    assert classify_severity(percent_change=60.0, cohen_d=1.5, p_value=1e-5) == "high"
    assert classify_severity(percent_change=150.0, cohen_d=3.5, p_value=1e-12) == "critical"


# ---------------------------------------------------------------------------
# Test Metric Shift Analysis
# ---------------------------------------------------------------------------
def test_analyze_metric_shift_zero_baseline():
    before = pd.Series([0.0, 0.0, 0.0, 0.0])
    after = pd.Series([5.0, 5.0, 5.0, 5.0])
    impact = analyze_metric_shift(before, after, "failed_requests")

    assert impact.percent_change is None  # Handled division-by-zero cleanly
    assert impact.absolute_change == 5.0
    assert impact.metric == "failed_requests"


# ---------------------------------------------------------------------------
# Test Real Dataset Impact Analysis
# ---------------------------------------------------------------------------
def test_analyze_change_impact_sample_dataset():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")
    report = analyze_change_impact(df, change_index=72)

    assert isinstance(report, ChangeImpactReport)
    assert report.change_index == 72
    assert report.timestamp == pd.to_datetime("2026-01-01 15:00:00")
    assert "response_time_ms" in report.metrics
    assert "failed_requests" in report.metrics

    resp_impact = report.metrics["response_time_ms"]
    assert resp_impact.absolute_change > 200.0
    assert resp_impact.percent_change is not None and resp_impact.percent_change > 100.0
    assert resp_impact.cohen_d > 5.0
    assert resp_impact.is_statistically_significant is True
    assert resp_impact.p_value is not None and resp_impact.p_value < 1e-10
    assert resp_impact.severity == "critical"
    assert report.overall_severity == "critical"

    # Test serialization
    data = report.to_dict()
    assert data["change_index"] == 72
    assert data["overall_severity"] == "critical"
    assert "response_time_ms" in data["metrics"]
    assert data["metrics"]["response_time_ms"]["severity"] == "critical"


def test_analyze_change_impact_with_window_size():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")
    # Compare only 15 points before vs 15 points after index 72
    report = analyze_change_impact(df, change_index=72, window_size=15)

    assert report.window_size == 15
    resp_impact = report.metrics["response_time_ms"]
    assert resp_impact.before_stats.count == 15
    assert resp_impact.after_stats.count == 15
    assert resp_impact.is_statistically_significant is True


def test_analyze_change_impact_no_change_dataset():
    df = pd.read_csv(DATA_DIR / "no_change_metrics.csv")
    report = analyze_change_impact(df, change_index=60)

    resp_impact = report.metrics["response_time_ms"]
    assert resp_impact.is_statistically_significant is False
    assert resp_impact.severity == "negligible"
    assert abs(resp_impact.percent_change) < 2.0


def test_analyze_change_impact_invalid_indices():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")

    with pytest.raises(ValueError, match="change_index"):
        analyze_change_impact(df, change_index=0)

    with pytest.raises(ValueError, match="change_index"):
        analyze_change_impact(df, change_index=len(df))


def test_analyze_all_changes_batch():
    df = pd.read_csv(DATA_DIR / "sample_api_metrics.csv")
    reports = analyze_all_changes(df, change_indices=[72, 95])

    assert len(reports) == 2
    assert reports[0].change_index == 72
    assert reports[1].change_index == 95
