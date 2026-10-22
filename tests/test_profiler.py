"""
Unit tests for the Per-Endpoint API Profiler module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from patterntime.profiler import (
    EndpointProfile,
    ProfileReport,
    find_grouping_columns,
    find_status_column,
    profile_endpoints,
)


@pytest.fixture
def sample_api_df():
    """Create a synthetic DataFrame simulating multiple API endpoints."""
    np.random.seed(42)
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=100, freq="1min")

    # Endpoint A: Stable 50ms latency
    df_a = pd.DataFrame({
        "timestamp": timestamps[:50],
        "api_name": "GET /api/v1/users",
        "latency_ms": np.random.normal(50, 5, 50),
        "status_code": [200] * 48 + [500, 502],
    })

    # Endpoint B: Steps up from 100ms to 350ms at index 25
    latencies_b = np.concatenate([
        np.random.normal(100, 10, 25),
        np.random.normal(350, 15, 25),
    ])
    df_b = pd.DataFrame({
        "timestamp": timestamps[50:],
        "api_name": "POST /api/v1/orders",
        "latency_ms": latencies_b,
        "status_code": [200] * 50,
    })

    return pd.concat([df_a, df_b], ignore_index=True)


def test_find_grouping_columns(sample_api_df):
    """Test auto-discovery of grouping columns."""
    cols = find_grouping_columns(sample_api_df)
    assert "api_name" in cols


def test_find_status_column(sample_api_df):
    """Test auto-discovery of status code columns."""
    status_col = find_status_column(sample_api_df)
    assert status_col == "status_code"


def test_profile_endpoints_synthetic(sample_api_df):
    """Test per-endpoint profiling and regression isolation on synthetic data."""
    report = profile_endpoints(
        sample_api_df,
        group_col="api_name",
        metric_col="latency_ms",
        status_col="status_code",
        timestamp_col="timestamp",
        algorithm="sliding_window",
        min_samples=15,
    )

    assert isinstance(report, ProfileReport)
    assert report.total_endpoints == 2
    assert report.total_requests == 100

    # Profile dictionary mapping
    profiles = {ep.endpoint_name: ep for ep in report.endpoints}

    # Verify Endpoint A
    ep_a = profiles["GET /api/v1/users"]
    assert ep_a.request_count == 50
    assert ep_a.error_count == 2
    assert ep_a.error_rate_pct == 4.0
    assert 45 < ep_a.mean_latency < 55

    # Verify Endpoint B (Regression detected)
    ep_b = profiles["POST /api/v1/orders"]
    assert ep_b.request_count == 50
    assert ep_b.error_count == 0
    assert ep_b.error_rate_pct == 0.0
    assert ep_b.total_events > 0
    assert ep_b.max_cohen_d > 5.0
    assert ep_b.max_severity in ["critical", "high", "medium"]

    # Regressed endpoints ranking
    assert len(report.regressed_endpoints) >= 1
    assert report.regressed_endpoints[0].endpoint_name == "POST /api/v1/orders"


def test_profile_endpoints_real_cakephp():
    """Test profiling on real-world CakePHP API logs (data/user_activities.csv)."""
    csv_path = Path("data/user_activities.csv")
    if not csv_path.exists():
        pytest.skip("data/user_activities.csv not available")

    df = pd.read_csv(csv_path)
    report = profile_endpoints(
        df,
        group_col="api_name",
        metric_col="execution_time_ms",
        status_col="status_code",
        timestamp_col="timestamp",
        algorithm="ewma",
        min_samples=30,
        dataset_name="user_activities.csv",
    )

    assert report.total_requests == 9081
    assert report.total_endpoints > 10

    # Serialization test
    rep_dict = report.to_dict()
    assert "regressed_endpoints" in rep_dict
    assert rep_dict["regressed_count"] >= 1

    top_regressed = report.regressed_endpoints[0]
    assert top_regressed.request_count >= 30
    assert top_regressed.total_events > 0


def test_profile_endpoints_edge_cases():
    """Test error handling for empty DataFrames and invalid columns."""
    with pytest.raises(ValueError, match="DataFrame is empty"):
        profile_endpoints(pd.DataFrame())

    df = pd.DataFrame({"col1": [1, 2, 3]})
    with pytest.raises(ValueError, match="No categorical column"):
        profile_endpoints(df, group_col=None)
