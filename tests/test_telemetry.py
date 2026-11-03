"""
Unit tests for the PatternTime Telemetry Ingestion Engine.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from patterntime.telemetry import (
    get_telemetry_slice,
    ingest_telemetry_batch,
    list_telemetry_datasets,
    parse_telemetry_payload,
    sanitize_dataset_name,
    validate_and_prepare_records,
)
from patterntime.storage import DatabaseStorage


def test_sanitize_dataset_name():
    """Test sanitization against path traversal and malformed inputs."""
    assert sanitize_dataset_name("../../etc/passwd") == "passwd.csv"
    assert sanitize_dataset_name("..\\windows\\system32\\cmd.exe") == "cmd.exe.csv"
    assert sanitize_dataset_name("my-telemetry") == "my-telemetry.csv"
    assert sanitize_dataset_name("app_metrics.csv") == "app_metrics.csv"
    assert sanitize_dataset_name("   ") == "api_telemetry.csv"
    assert sanitize_dataset_name(None) == "api_telemetry.csv"


def test_parse_telemetry_payload_formats():
    """Test parsing batch object, list of dicts, and single dict."""
    # Format 1: Envelope with records
    p1 = {
        "dataset": "gateway.csv",
        "service": "api-gateway",
        "mode": "overwrite",
        "detect": True,
        "records": [{"timestamp": "2026-11-01 10:00:00", "latency": 50}],
    }
    ds1, recs1, opts1 = parse_telemetry_payload(p1)
    assert ds1 == "gateway.csv"
    assert len(recs1) == 1
    assert opts1["service"] == "api-gateway"
    assert opts1["mode"] == "overwrite"
    assert opts1["detect"] is True

    # Format 2: Raw list
    p2 = [
        {"timestamp": "2026-11-01 10:00:00", "latency": 50},
        {"timestamp": "2026-11-01 10:00:05", "latency": 55},
    ]
    ds2, recs2, opts2 = parse_telemetry_payload(p2)
    assert ds2 == "api_telemetry.csv"
    assert len(recs2) == 2
    assert opts2["mode"] == "append"

    # Format 3: Single record
    p3 = {"timestamp": "2026-11-01 10:00:00", "latency": 50}
    ds3, recs3, _ = parse_telemetry_payload(p3)
    assert ds3 == "api_telemetry.csv"
    assert len(recs3) == 1

    # Empty payload errors
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_telemetry_payload([])

    with pytest.raises(ValueError, match="cannot be empty"):
        parse_telemetry_payload({"records": []})


def test_validate_and_prepare_records():
    """Test validation and DataFrame normalization."""
    records = [
        {"timestamp": "2026-11-01 12:00:00", "route": "/login", "latency_ms": 120.5, "status": 200},
        {"timestamp": "2026-11-01 12:00:05", "route": "/login", "latency_ms": 115.0, "status": 200},
    ]
    df, ts_col, metrics = validate_and_prepare_records(records, service_tag="auth-svc")

    assert ts_col == "timestamp"
    assert "latency_ms" in metrics
    assert "status" in metrics
    assert "service" in df.columns
    assert df["service"].iloc[0] == "auth-svc"
    assert len(df) == 2

    # Auto-generate timestamp if missing
    no_ts = [{"latency_ms": 100}, {"latency_ms": 105}]
    df_no_ts, ts_gen, _ = validate_and_prepare_records(no_ts)
    assert ts_gen == "timestamp"
    assert "timestamp" in df_no_ts.columns

    # Reject records without numeric metrics
    text_only = [{"timestamp": "2026-11-01 12:00:00", "message": "hello"}]
    with pytest.raises(ValueError, match="No numeric metric columns"):
        validate_and_prepare_records(text_only)


def test_ingest_telemetry_batch_append_and_slice(tmp_path):
    """Test file creation, appending with column alignment, and slicing."""
    data_dir = tmp_path / "data"

    # Batch 1: Initial creation
    batch1 = {
        "dataset": "test_app.csv",
        "service": "checkout",
        "records": [
            {"timestamp": "2026-11-01 10:00:00", "route": "/pay", "duration_ms": 150.0},
            {"timestamp": "2026-11-01 10:00:05", "route": "/pay", "duration_ms": 155.0},
        ],
    }

    res1 = ingest_telemetry_batch(batch1, data_dir=data_dir)
    assert res1["status"] == "success"
    assert res1["dataset"] == "test_app.csv"
    assert res1["records_ingested"] == 2
    assert res1["total_rows"] == 2

    # Batch 2: Append with new column
    batch2 = {
        "dataset": "test_app.csv",
        "records": [
            {"timestamp": "2026-11-01 10:00:10", "route": "/refund", "duration_ms": 200.0, "memory_mb": 45.0},
            {"timestamp": "2026-11-01 10:00:15", "route": "/pay", "duration_ms": 160.0, "memory_mb": 42.0},
        ],
    }
    res2 = ingest_telemetry_batch(batch2, data_dir=data_dir)
    assert res2["records_ingested"] == 2
    assert res2["total_rows"] == 4

    # Test get_telemetry_slice
    slice_all = get_telemetry_slice("test_app.csv", data_dir=data_dir, limit=10)
    assert slice_all["total_rows"] == 4
    assert len(slice_all["records"]) == 4
    assert "memory_mb" in slice_all["columns"]

    # Test route filtering
    slice_filtered = get_telemetry_slice("test_app.csv", data_dir=data_dir, route_filter="/refund")
    assert slice_filtered["filtered_rows"] == 1
    assert slice_filtered["records"][0]["route"] == "/refund"


def test_ingest_telemetry_with_auto_detect(tmp_path):
    """Test on-the-fly change detection when ingest option detect=True."""
    data_dir = tmp_path / "data"

    # Generate 40 baseline records and 20 shifted records
    records = []
    for i in range(40):
        records.append({
            "timestamp": f"2026-11-01 10:{i:02d}:00",
            "latency": 50.0 + (i % 3),
        })
    for i in range(40, 60):
        records.append({
            "timestamp": f"2026-11-01 10:{i:02d}:00",
            "latency": 120.0 + (i % 3),
        })

    payload = {
        "dataset": "perf_test.csv",
        "detect": True,
        "algorithm": "ewma",
        "metric": "latency",
        "records": records,
    }

    res = ingest_telemetry_batch(payload, data_dir=data_dir)
    assert res["status"] == "success"
    assert "detection" in res
    det = res["detection"]
    assert det["events_count"] > 0
    assert "severity" in det


def test_list_telemetry_datasets(tmp_path):
    """Test enumeration of datasets in data_dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy CSV files
    df1 = pd.DataFrame({"timestamp": ["2026-11-01 00:00:00"], "latency": [10.0]})
    df1.to_csv(data_dir / "ds1.csv", index=False)

    df2 = pd.DataFrame({"timestamp": ["2026-11-01 00:00:00", "2026-11-01 00:01:00"], "cpu": [20.0, 25.0]})
    df2.to_csv(data_dir / "ds2.csv", index=False)

    datasets = list_telemetry_datasets(data_dir=data_dir)
    assert len(datasets) == 2
    names = [d["name"] for d in datasets]
    assert "ds1.csv" in names
    assert "ds2.csv" in names
