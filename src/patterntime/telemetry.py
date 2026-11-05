"""
Telemetry Ingestion & Management Engine for PatternTime.

Provides ingestion, validation, persistence, and querying of time-series
performance telemetry batches emitted by external applications, services, and APIs.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from patterntime.detectors import (
    ChangeEvent,
    DetectionResult,
    get_detector,
)
from patterntime.impact import analyze_change_impact
from patterntime.storage import DatabaseStorage
from patterntime.validator import (
    COMMON_EXCLUDED_METRICS,
    COMMON_TIMESTAMP_NAMES,
    MissingTimestampError,
    NoNumericMetricsError,
    detect_metric_columns,
    detect_timestamp_column,
    infer_time_frequency,
)

# Global re-entrant lock to ensure thread-safe CSV append operations
_FILE_LOCK = threading.RLock()


def sanitize_dataset_name(name: str | None) -> str:
    """
    Sanitize and normalize dataset file names to prevent path traversal
    and ensure safety across file systems.
    """
    if not name or not isinstance(name, str) or not name.strip():
        return "api_telemetry.csv"

    # Normalize backslashes for cross-platform compatibility and strip directory paths
    normalized = name.strip().replace("\\", "/")
    base = os.path.basename(normalized)
    # Strip dangerous characters
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    clean = clean.lstrip(".")

    if not clean:
        clean = "api_telemetry"

    if not clean.lower().endswith(".csv"):
        clean = f"{clean}.csv"

    return clean


def parse_telemetry_payload(
    payload: Any,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """
    Normalize different payload formats into a consistent tuple:
    (dataset_name, records, options)

    Supported formats:
    1. Batch object:
       {"dataset": "...", "service": "...", "mode": "append", "records": [...], "detect": true}
    2. List of record objects:
       [{"timestamp": "...", "latency_ms": 120}, ...]
    3. Single record object:
       {"timestamp": "...", "latency_ms": 120}
    """
    options: dict[str, Any] = {
        "mode": "append",
        "detect": False,
        "algorithm": "ewma",
        "metric": None,
        "service": None,
        "application": None,
        "environment": None,
    }

    if payload is None:
        raise ValueError("Payload cannot be empty.")

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Telemetry records list cannot be empty.")
        records = payload
        dataset_name = "api_telemetry.csv"
        return dataset_name, records, options

    if isinstance(payload, dict):
        # Check if this is an envelope containing records list
        records_candidate = None
        for key in ("records", "data", "metrics", "points", "events"):
            if key in payload and isinstance(payload[key], list):
                records_candidate = payload[key]
                break

        if records_candidate is not None:
            if not records_candidate:
                raise ValueError("Telemetry records list cannot be empty.")
            records = records_candidate
            dataset_name = sanitize_dataset_name(payload.get("dataset"))
            options["mode"] = str(payload.get("mode", "append")).lower()
            options["detect"] = bool(payload.get("detect", False))
            options["algorithm"] = str(payload.get("algorithm", "ewma")).lower()
            options["metric"] = payload.get("metric")
            options["service"] = payload.get("service")
            options["application"] = payload.get("application")
            options["environment"] = payload.get("environment")
            return dataset_name, records, options

        # Single record dictionary
        if any(key in payload for key in COMMON_TIMESTAMP_NAMES) or any(
            isinstance(v, (int, float)) for v in payload.values()
        ):
            options["application"] = payload.get("application")
            options["environment"] = payload.get("environment")
            record = {
                key: value
                for key, value in payload.items()
                if key not in ("application", "environment", "dataset")
            }
            records = [record]
            dataset_name = sanitize_dataset_name(payload.get("dataset"))
            return dataset_name, records, options

    raise ValueError(
        "Invalid telemetry payload structure. Expected an object with 'records' array or a list of record objects."
    )


def validate_and_prepare_records(
    records: list[dict[str, Any]],
    service_tag: str | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Validate telemetry records and convert into a normalized DataFrame.
    Returns: (DataFrame, timestamp_col, metric_cols)
    """
    if not records:
        raise ValueError("No telemetry records provided.")

    df = pd.DataFrame(records)

    # Attach optional service tag if provided and not already in columns
    if service_tag and "service" not in df.columns:
        df["service"] = str(service_tag)

    # Detect or auto-fill timestamp
    timestamp_col = None
    try:
        timestamp_col = detect_timestamp_column(df)
    except MissingTimestampError:
        # If timestamp is missing in live telemetry, generate UTC timestamps
        now_iso = datetime.now(timezone.utc).isoformat()
        df["timestamp"] = now_iso
        timestamp_col = "timestamp"

    # Coerce timestamp column to string ISO format
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        if df[timestamp_col].isna().all():
            raise ValueError(f"All timestamps in column '{timestamp_col}' failed datetime conversion.")
        # Fill any individual invalid timestamps with forward fill / now
        df[timestamp_col] = df[timestamp_col].ffill().bfill().dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as err:
        raise ValueError(f"Failed parsing timestamps in '{timestamp_col}': {err}")

    # Detect numeric metric columns
    try:
        metric_cols = detect_metric_columns(df, timestamp_col=timestamp_col, exclude_flags=True)
    except NoNumericMetricsError:
        raise ValueError(
            f"No numeric metric columns found in telemetry records. Columns present: {list(df.columns)}. "
            "At least one numeric measurement (e.g. duration_ms, latency, cpu, memory_mb) is required."
        )

    # Ensure numeric types are cast cleanly
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, timestamp_col, metric_cols


def ingest_telemetry_batch(
    payload: Any,
    data_dir: Path | str = "data",
    storage: DatabaseStorage | None = None,
) -> dict[str, Any]:
    """
    Process, validate, persist, and optionally run detection on a batch of telemetry.

    Parameters:
        payload: The JSON-decoded payload (batch object, record list, or single record).
        data_dir: Target directory where CSV datasets are stored (defaults to 'data').
        storage: Optional DatabaseStorage instance for MySQL metadata persistence.

    Returns:
        Summary dict containing ingestion metrics, total rows, dataset info, and optional detection.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    dataset_name, records, options = parse_telemetry_payload(payload)
    df_new, timestamp_col, metric_cols = validate_and_prepare_records(
        records, service_tag=options.get("service")
    )

    csv_file = data_path / dataset_name
    mode = options.get("mode", "append")

    with _FILE_LOCK:
        if csv_file.exists() and mode == "append" and csv_file.stat().st_size > 0:
            # Read existing file to align columns
            try:
                df_existing = pd.read_csv(csv_file)
                # Align columns
                all_cols = list(dict.fromkeys(list(df_existing.columns) + list(df_new.columns)))
                df_existing = df_existing.reindex(columns=all_cols)
                df_new = df_new.reindex(columns=all_cols)

                # Concatenate and write back
                combined_df = pd.concat([df_existing, df_new], ignore_index=True)
                combined_df.to_csv(csv_file, index=False)
                total_rows = len(combined_df)
                active_df = combined_df
            except Exception:
                # If reading existing CSV fails, append directly
                df_new.to_csv(csv_file, mode="a", header=False, index=False)
                total_rows = sum(1 for _ in open(csv_file, "r", encoding="utf-8")) - 1
                active_df = df_new
        else:
            # Write new file or overwrite
            df_new.to_csv(csv_file, index=False)
            total_rows = len(df_new)
            active_df = df_new

    # Infer frequency / cadence
    cadence = None
    try:
        ts_series = pd.to_datetime(active_df[timestamp_col], errors="coerce").dropna()
        cadence = infer_time_frequency(ts_series)
    except Exception:
        pass

    # Synchronize with DatabaseStorage if available
    dataset_id = None
    if storage is not None:
        try:
            dataset_id = storage.save_dataset(
                name=dataset_name,
                row_count=total_rows,
                cadence=cadence,
                timestamp_col=timestamp_col,
                metric_cols=metric_cols,
                application=options.get("application"),
                environment=options.get("environment"),
            )
        except Exception:
            # MySQL persistence failure should not block successful telemetry ingestion
            pass

    response: dict[str, Any] = {
        "status": "success",
        "message": f"Ingested {len(records)} telemetry records into '{dataset_name}'",
        "dataset": dataset_name,
        "records_ingested": len(records),
        "total_rows": total_rows,
        "metrics": metric_cols,
        "timestamp_col": timestamp_col,
        "cadence": cadence,
        "dataset_id": dataset_id,
    }

    # Optional On-the-Fly Change Detection
    if options.get("detect") and total_rows >= 10:
        target_metric = options.get("metric") or metric_cols[0]
        algorithm_name = options.get("algorithm") or "ewma"

        try:
            detector = get_detector(algorithm_name)
            series = pd.to_numeric(active_df[target_metric], errors="coerce").fillna(0.0)
            timestamps = active_df[timestamp_col] if timestamp_col in active_df.columns else None

            det_result = detector.detect(series, timestamps=timestamps)

            detection_summary: dict[str, Any] = {
                "algorithm": algorithm_name,
                "metric": target_metric,
                "events_count": len(det_result.events),
                "execution_time_ms": round(det_result.execution_time_ms, 2),
                "events": [e.to_dict() for e in det_result.events],
            }

            if det_result.has_changes:
                # Compute statistical impact of the most significant change event
                first_event = det_result.events[0]
                impact_report = analyze_change_impact(
                    df=active_df,
                    change_index=first_event.index,
                    metric_cols=[target_metric],
                    timestamp_col=timestamp_col,
                )
                metric_impact = impact_report.metrics.get(target_metric)
                if metric_impact:
                    detection_summary["impact"] = metric_impact.to_dict()
                    detection_summary["severity"] = metric_impact.severity
                else:
                    detection_summary["severity"] = impact_report.overall_severity

                # Persist detection run if storage is configured
                if storage is not None and dataset_id is not None:
                    try:
                        run_id = storage.save_run(
                            dataset_name=dataset_name,
                            metric_name=target_metric,
                            detection_result=det_result,
                            impact_report=metric_impact or impact_report.to_dict(),
                            dataset_id=dataset_id,
                        )
                        detection_summary["run_id"] = run_id
                    except Exception:
                        pass

            response["detection"] = detection_summary
        except Exception as err:
            response["detection_warning"] = f"Automatic change detection failed: {err}"

    return response


def get_telemetry_slice(
    dataset_name: str,
    data_dir: Path | str = "data",
    limit: int = 50,
    offset: int = 0,
    route_filter: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve sliced records and metadata from an ingested CSV telemetry dataset.
    """
    clean_name = sanitize_dataset_name(dataset_name)
    csv_file = Path(data_dir) / clean_name

    if not csv_file.exists():
        raise FileNotFoundError(f"Telemetry dataset '{clean_name}' not found.")

    with _FILE_LOCK:
        df = pd.read_csv(csv_file)

    total_rows = len(df)
    columns = list(df.columns)

    # Filter by route if requested and route column exists
    route_col = None
    for candidate in ("route", "endpoint", "path", "url", "action"):
        if candidate in df.columns:
            route_col = candidate
            break

    if route_filter and route_col:
        df = df[df[route_col].astype(str) == str(route_filter)]

    filtered_rows = len(df)

    # Detect timestamp and metrics
    timestamp_col = None
    try:
        timestamp_col = detect_timestamp_column(df)
    except Exception:
        pass

    metric_cols = []
    if timestamp_col:
        try:
            metric_cols = detect_metric_columns(df, timestamp_col=timestamp_col, exclude_flags=True)
        except Exception:
            pass

    time_range = None
    cadence = None
    if timestamp_col and not df.empty:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
            if not ts.empty:
                time_range = {
                    "start": ts.min().isoformat(),
                    "end": ts.max().isoformat(),
                }
                cadence = infer_time_frequency(ts)
        except Exception:
            pass

    # Apply pagination
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    slice_df = df.iloc[offset : offset + limit]

    # Convert records to JSON-safe python dicts (replace NaN with None)
    records = slice_df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "dataset": clean_name,
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "limit": limit,
        "offset": offset,
        "columns": columns,
        "metrics": metric_cols,
        "timestamp_col": timestamp_col,
        "cadence": cadence,
        "time_range": time_range,
        "records": records,
    }


def list_telemetry_datasets(data_dir: Path | str = "data") -> list[dict[str, Any]]:
    """
    Enumerate all CSV telemetry datasets present in the data directory.
    """
    path = Path(data_dir)
    if not path.exists():
        return []

    results = []
    for f in sorted(path.glob("*.csv")):
        try:
            stat = f.stat()
            # Quickly sample first row for columns and count lines
            df_head = pd.read_csv(f, nrows=5)
            row_count = sum(1 for _ in open(f, "r", encoding="utf-8", errors="ignore")) - 1

            ts_col = None
            try:
                ts_col = detect_timestamp_column(df_head)
            except Exception:
                pass

            metrics = []
            if ts_col:
                try:
                    metrics = detect_metric_columns(df_head, timestamp_col=ts_col)
                except Exception:
                    pass

            results.append({
                "name": f.name,
                "row_count": max(0, row_count),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "timestamp_col": ts_col,
                "metrics": metrics,
            })
        except Exception:
            continue

    return results
