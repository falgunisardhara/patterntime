from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class TimeSeriesValidationError(Exception):
    """Base exception for all time-series validation errors."""


class EmptyDataError(TimeSeriesValidationError):
    """Raised when the input time-series data is empty."""


class InsufficientDataError(TimeSeriesValidationError):
    """Raised when there are fewer data points than required."""


class MissingTimestampError(TimeSeriesValidationError):
    """Raised when no valid timestamp column could be found or specified."""


class InvalidTimestampError(TimeSeriesValidationError):
    """Raised when timestamps cannot be parsed or contain invalid dates."""


class DuplicateTimestampError(TimeSeriesValidationError):
    """Raised when duplicate timestamps are present in the time series."""


class MissingValueError(TimeSeriesValidationError):
    """Raised when NaN or null values are encountered in metric columns."""


class ZeroVarianceError(TimeSeriesValidationError):
    """Raised when a metric column has zero variance (constant values)."""


class NoNumericMetricsError(TimeSeriesValidationError):
    """Raised when no numeric metric columns are found for analysis."""


# ---------------------------------------------------------------------------
# Configuration & Result Dataclasses
# ---------------------------------------------------------------------------
COMMON_TIMESTAMP_NAMES = [
    "timestamp",
    "datetime",
    "date",
    "time",
    "ts",
    "recorded_at",
    "created_at",
]

COMMON_EXCLUDED_METRICS = [
    "is_after_change",
    "label",
    "anomaly",
    "id",
]


@dataclass
class ValidationConfig:
    """Configuration options for time-series ingestion and validation."""

    timestamp_col: str | None = None
    metric_cols: list[str] | None = None
    min_points: int = 10
    missing_value_strategy: str = "error"  # "error", "drop", "interpolate", "ffill"
    handle_duplicates: str = "error"  # "error", "keep_first", "keep_last", "average"
    sort_by_time: bool = True
    require_non_zero_variance: bool = True
    exclude_flag_columns: bool = True


@dataclass
class ValidationResult:
    """Structured report returned after successfully validating a time series."""

    df: pd.DataFrame
    timestamp_col: str
    metric_cols: list[str]
    inferred_frequency: str | None
    time_range: tuple[pd.Timestamp, pd.Timestamp]
    total_rows: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation summary to a JSON-serializable dictionary."""
        return {
            "timestamp_col": self.timestamp_col,
            "metric_cols": self.metric_cols,
            "inferred_frequency": self.inferred_frequency,
            "start_time": self.time_range[0].isoformat() if self.time_range else None,
            "end_time": self.time_range[1].isoformat() if self.time_range else None,
            "total_rows": self.total_rows,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Core Validator Functions
# ---------------------------------------------------------------------------
def detect_timestamp_column(df: pd.DataFrame, explicit_col: str | None = None) -> str:
    """
    Identify the timestamp column in a DataFrame:
    1. If explicit_col is given, check that it exists.
    2. Otherwise, check common timestamp column names (case-insensitive).
    3. If still not found, try parsing string/object columns to find a datetime column.
    """
    if explicit_col is not None:
        if explicit_col not in df.columns:
            raise MissingTimestampError(
                f"Specified timestamp column '{explicit_col}' not found in columns: {list(df.columns)}"
            )
        return explicit_col

    lower_cols = {str(col).lower(): col for col in df.columns}
    for candidate in COMMON_TIMESTAMP_NAMES:
        if candidate in lower_cols:
            return lower_cols[candidate]

    # Search for an object/string column that parses into valid datetimes
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(15)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        return str(col)
                except Exception:
                    continue

    raise MissingTimestampError(
        f"Could not automatically detect a timestamp column in columns: {list(df.columns)}. "
        "Please specify timestamp_col explicitly."
    )


def detect_metric_columns(
    df: pd.DataFrame,
    timestamp_col: str,
    explicit_metrics: list[str] | None = None,
    exclude_flags: bool = True,
) -> list[str]:
    """
    Identify numeric columns suitable for time-series analysis.
    """
    if explicit_metrics is not None:
        missing = [col for col in explicit_metrics if col not in df.columns]
        if missing:
            raise NoNumericMetricsError(
                f"Specified metric columns not found in dataset: {missing}"
            )
        non_numeric = [
            col for col in explicit_metrics if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise NoNumericMetricsError(
                f"Specified metric columns are not numeric: {non_numeric}"
            )
        return explicit_metrics

    numeric_cols: list[str] = []
    for col in df.columns:
        if col == timestamp_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if exclude_flags and str(col).lower() in COMMON_EXCLUDED_METRICS:
                continue
            numeric_cols.append(str(col))

    if not numeric_cols:
        raise NoNumericMetricsError(
            f"No numeric metric columns found in dataset (columns: {list(df.columns)})"
        )

    return numeric_cols


def infer_time_frequency(series: pd.Series) -> str | None:
    """
    Infer the regular cadence of the time series, returning a human-readable interval
    string (e.g. '5min', '1h', '1D') or None if irregular.
    """
    if len(series) < 3:
        return None

    # Try standard pandas frequency inference
    try:
        inferred = pd.infer_freq(pd.DatetimeIndex(series))
        if inferred is not None:
            # Normalize pandas frequency aliases (e.g. 'h' -> '1h', 'min' -> '1min')
            if inferred in {"h", "min", "s", "D", "W", "M", "Y"}:
                return f"1{inferred.lower()}"
            return inferred.lower()
    except Exception:
        pass

    # Fallback: inspect median time delta
    deltas = series.diff().dropna()
    if deltas.empty:
        return None

    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        return None

    # If >80% of steps match the median delta, return formatted interval
    matches = (deltas == median_delta).mean()
    if matches >= 0.8:
        total_seconds = int(median_delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        if total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes}min"
        if total_seconds < 86400:
            hours = total_seconds // 3600
            return f"{hours}h"
        days = total_seconds // 86400
        return f"{days}D"

    return None


def validate_dataframe(
    df: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """
    Validate, sanitize, and prepare a pandas DataFrame for time-series analysis.
    """
    if config is None:
        config = ValidationConfig()

    warnings: list[str] = []

    # 1. Check for empty dataframe
    if df.empty:
        raise EmptyDataError("Input dataset is empty (0 rows).")

    working_df = df.copy()

    # 2. Identify timestamp column
    ts_col = detect_timestamp_column(working_df, config.timestamp_col)

    # 3. Parse timestamps to datetime
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            working_df[ts_col] = pd.to_datetime(working_df[ts_col], errors="coerce")
    except Exception as exc:
        raise InvalidTimestampError(
            f"Failed to parse timestamp column '{ts_col}': {exc}"
        ) from exc

    invalid_ts_count = working_df[ts_col].isna().sum()
    if invalid_ts_count > 0:
        raise InvalidTimestampError(
            f"Timestamp column '{ts_col}' contains {invalid_ts_count} invalid or unparseable date values."
        )

    # 4. Handle duplicates in timestamps
    duplicate_count = working_df.duplicated(subset=[ts_col]).sum()
    if duplicate_count > 0:
        if config.handle_duplicates == "error":
            raise DuplicateTimestampError(
                f"Found {duplicate_count} duplicate timestamps in column '{ts_col}'."
            )
        if config.handle_duplicates == "keep_first":
            working_df = working_df.drop_duplicates(subset=[ts_col], keep="first")
            warnings.append(f"Dropped {duplicate_count} duplicate timestamps (kept first).")
        elif config.handle_duplicates == "keep_last":
            working_df = working_df.drop_duplicates(subset=[ts_col], keep="last")
            warnings.append(f"Dropped {duplicate_count} duplicate timestamps (kept last).")
        elif config.handle_duplicates == "average":
            # Group by timestamp and take mean of numeric columns
            num_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            working_df = working_df.groupby(ts_col, as_index=False)[num_cols].mean()
            warnings.append(f"Averaged values for {duplicate_count} duplicate timestamps.")
        else:
            raise ValueError(f"Unknown handle_duplicates option: {config.handle_duplicates}")

    # 5. Chronological sorting
    if config.sort_by_time:
        if not working_df[ts_col].is_monotonic_increasing:
            working_df = working_df.sort_values(by=ts_col).reset_index(drop=True)
            warnings.append("Timestamps were out of chronological order and have been sorted.")

    # 6. Check row count against minimum points
    if len(working_df) < config.min_points:
        raise InsufficientDataError(
            f"Dataset has {len(working_df)} rows, but at least {config.min_points} points are required."
        )

    # 7. Identify metric columns
    metric_cols = detect_metric_columns(
        working_df,
        timestamp_col=ts_col,
        explicit_metrics=config.metric_cols,
        exclude_flags=config.exclude_flag_columns,
    )

    # 8. Handle missing / NaN values in metrics
    for metric in metric_cols:
        nan_count = working_df[metric].isna().sum()
        if nan_count > 0:
            if config.missing_value_strategy == "error":
                raise MissingValueError(
                    f"Metric column '{metric}' contains {nan_count} missing (NaN) values."
                )
            if config.missing_value_strategy == "drop":
                working_df = working_df.dropna(subset=[metric]).reset_index(drop=True)
                warnings.append(f"Dropped {nan_count} rows with missing values in '{metric}'.")
            elif config.missing_value_strategy == "interpolate":
                working_df[metric] = (
                    working_df[metric].interpolate(method="linear").bfill().ffill()
                )
                warnings.append(f"Linearly interpolated {nan_count} missing values in '{metric}'.")
            elif config.missing_value_strategy == "ffill":
                working_df[metric] = working_df[metric].ffill().bfill()
                warnings.append(f"Forward-filled {nan_count} missing values in '{metric}'.")
            else:
                raise ValueError(
                    f"Unknown missing_value_strategy: {config.missing_value_strategy}"
                )

    # Re-check row count in case drop strategy removed points
    if len(working_df) < config.min_points:
        raise InsufficientDataError(
            f"Dataset has {len(working_df)} rows after cleaning, fewer than required {config.min_points}."
        )

    # 9. Verify non-zero variance (avoid flat-line metrics breaking Z-score detectors)
    if config.require_non_zero_variance:
        for metric in metric_cols:
            std_dev = float(working_df[metric].std(ddof=1))
            if np.isclose(std_dev, 0.0) or working_df[metric].nunique() <= 1:
                raise ZeroVarianceError(
                    f"Metric column '{metric}' has zero variance (constant values). "
                    "Cannot compute standardization or detect shifts."
                )

    # 10. Infer frequency & time range
    inferred_freq = infer_time_frequency(working_df[ts_col])
    time_range = (
        working_df[ts_col].iloc[0],
        working_df[ts_col].iloc[-1],
    )

    return ValidationResult(
        df=working_df,
        timestamp_col=ts_col,
        metric_cols=metric_cols,
        inferred_frequency=inferred_freq,
        time_range=time_range,
        total_rows=len(working_df),
        warnings=warnings,
    )


def load_and_validate_csv(
    csv_path: str | Path,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """
    Read a CSV file from disk and validate it into a clean TimeSeriesData container.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        raw_df = pd.read_csv(path)
    except Exception as exc:
        raise TimeSeriesValidationError(f"Could not parse CSV file '{path}': {exc}") from exc

    return validate_dataframe(raw_df, config=config)
