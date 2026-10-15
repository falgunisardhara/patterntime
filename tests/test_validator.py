from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patterntime.validator import (
    DuplicateTimestampError,
    EmptyDataError,
    InsufficientDataError,
    InvalidTimestampError,
    MissingTimestampError,
    MissingValueError,
    NoNumericMetricsError,
    TimeSeriesValidationError,
    ValidationConfig,
    ZeroVarianceError,
    load_and_validate_csv,
    validate_dataframe,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Test Existing Datasets Ingestion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename",
    [
        "sample_api_metrics.csv",
        "no_change_metrics.csv",
        "change_size_30ms.csv",
        "scenario_high_noise.csv",
        "scenario_transient_spikes.csv",
        "scenario_gradual_drift.csv",
    ],
)
def test_load_existing_project_csvs(filename: str):
    csv_path = DATA_DIR / filename
    result = load_and_validate_csv(csv_path)

    assert result.total_rows >= 60
    assert result.timestamp_col == "timestamp"
    assert "response_time_ms" in result.metric_cols
    assert "is_after_change" not in result.metric_cols  # Flag column excluded
    assert result.inferred_frequency == "5min"
    assert result.time_range[0] < result.time_range[1]

    # Verify JSON serializability
    data_dict = result.to_dict()
    assert data_dict["total_rows"] == result.total_rows
    assert data_dict["inferred_frequency"] == "5min"


# ---------------------------------------------------------------------------
# Test Timestamp Auto-Detection & Parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("col_name", ["timestamp", "datetime", "Date", "TS", "created_at"])
def test_detect_timestamp_various_names(col_name: str):
    dates = pd.date_range("2026-01-01", periods=15, freq="1h")
    df = pd.DataFrame({
        col_name: dates.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_a": np.random.default_rng(42).normal(50, 5, 15),
    })

    result = validate_dataframe(df)
    assert result.timestamp_col == col_name
    assert pd.api.types.is_datetime64_any_dtype(result.df[col_name])
    assert result.inferred_frequency == "1h"


def test_missing_timestamp_raises_error():
    df = pd.DataFrame({
        "value_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "value_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    })

    with pytest.raises(MissingTimestampError, match="Could not automatically detect"):
        validate_dataframe(df)


def test_invalid_unparseable_timestamp_raises_error():
    df = pd.DataFrame({
        "timestamp": ["not-a-date", "2026-01-01", "2026-01-02"] * 4,
        "value": range(12),
    })

    with pytest.raises(InvalidTimestampError, match="invalid or unparseable"):
        validate_dataframe(df)


# ---------------------------------------------------------------------------
# Test Duplicates & Out-of-Order Sorting
# ---------------------------------------------------------------------------
def test_duplicate_timestamps_default_raises_error():
    dates = list(pd.date_range("2026-01-01", periods=10, freq="10min"))
    dates.append(dates[2])  # Add duplicate

    df = pd.DataFrame({
        "timestamp": dates,
        "value": np.linspace(10, 20, 11),
    })

    with pytest.raises(DuplicateTimestampError, match="duplicate timestamps"):
        validate_dataframe(df)


def test_duplicate_timestamps_keep_first():
    dates = list(pd.date_range("2026-01-01", periods=10, freq="10min"))
    dates.append(dates[0])  # Duplicate of first point

    df = pd.DataFrame({
        "timestamp": dates,
        "value": [100.0] + list(range(1, 10)) + [999.0],
    })

    config = ValidationConfig(handle_duplicates="keep_first", min_points=5)
    result = validate_dataframe(df, config=config)

    assert result.total_rows == 10
    assert result.df["value"].iloc[0] == 100.0
    assert any("Dropped 1 duplicate" in w for w in result.warnings)


def test_duplicate_timestamps_average():
    dates = list(pd.date_range("2026-01-01", periods=10, freq="10min"))
    dates.append(dates[0])

    df = pd.DataFrame({
        "timestamp": dates,
        "value": [10.0] + list(range(1, 10)) + [20.0],  # 10 and 20 at same timestamp
    })

    config = ValidationConfig(handle_duplicates="average", min_points=5)
    result = validate_dataframe(df, config=config)

    assert result.total_rows == 10
    assert result.df["value"].iloc[0] == 15.0  # Average of 10.0 and 20.0


def test_out_of_order_timestamps_are_sorted():
    dates = pd.date_range("2026-01-01", periods=12, freq="15min")
    shuffled_dates = dates[::-1]  # Reversed order

    df = pd.DataFrame({
        "timestamp": shuffled_dates,
        "value": range(12),
    })

    result = validate_dataframe(df)
    assert result.df["timestamp"].is_monotonic_increasing
    assert any("sorted" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test Metric Identification & Missing Values
# ---------------------------------------------------------------------------
def test_empty_data_raises_error():
    df = pd.DataFrame()
    with pytest.raises(EmptyDataError, match="empty"):
        validate_dataframe(df)


def test_insufficient_points_raises_error():
    dates = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"timestamp": dates, "value": [1, 2, 3, 4, 5]})

    with pytest.raises(InsufficientDataError, match="points are required"):
        validate_dataframe(df, config=ValidationConfig(min_points=10))


def test_no_numeric_metrics_raises_error():
    dates = pd.date_range("2026-01-01", periods=12, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "status": ["OK", "WARN", "FAIL"] * 4,
    })

    with pytest.raises(NoNumericMetricsError, match="No numeric metric columns"):
        validate_dataframe(df)


def test_missing_values_default_raises_error():
    dates = pd.date_range("2026-01-01", periods=12, freq="1h")
    values = [10.0, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
    df = pd.DataFrame({"timestamp": dates, "latency": values})

    with pytest.raises(MissingValueError, match="missing"):
        validate_dataframe(df)


def test_missing_values_interpolate():
    dates = pd.date_range("2026-01-01", periods=12, freq="1h")
    values = [10.0, 12.0, np.nan, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    df = pd.DataFrame({"timestamp": dates, "latency": values})

    config = ValidationConfig(missing_value_strategy="interpolate")
    result = validate_dataframe(df, config=config)

    assert result.df["latency"].isna().sum() == 0
    assert result.df["latency"].iloc[2] == 14.0  # Linear midpoint between 12 and 16


def test_zero_variance_raises_error():
    dates = pd.date_range("2026-01-01", periods=12, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "constant_metric": [42.0] * 12,
    })

    with pytest.raises(ZeroVarianceError, match="zero variance"):
        validate_dataframe(df)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_validate_csv("non_existent_file.csv")
