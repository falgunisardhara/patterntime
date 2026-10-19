"""
Unit tests for the DatabaseStorage persistence module.
Tests schema creation, dataset registration, detection run persistence,
event tracking, and impact report storage.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from patterntime.detectors import ChangeEvent, DetectionResult
from patterntime.impact import MetricImpact, SegmentStats
from patterntime.storage import (
    ChangeEventModel,
    DatabaseStorage,
    DatasetModel,
    DetectionRunModel,
    ImpactReportModel,
    get_db_url,
)


@pytest.fixture
def memory_storage():
    """Return an isolated in-memory DatabaseStorage instance."""
    return DatabaseStorage(db_url="sqlite:///:memory:", auto_create_tables=True)


def test_get_db_url_default(monkeypatch):
    """Test default MySQL URL generation from environment variables."""
    monkeypatch.setenv("DB_USER", "root")
    monkeypatch.setenv("DB_PASSWORD", "secret#pass")
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_NAME", "patterntime")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    url = get_db_url()
    assert "mysql+pymysql://root:secret%23pass@127.0.0.1:3306/patterntime" in url


def test_get_db_url_override(monkeypatch):
    """Test that explicit DATABASE_URL takes precedence."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test_custom.db")
    url = get_db_url()
    assert url == "sqlite:///test_custom.db"


def test_test_connection(memory_storage):
    """Test connectivity check helper."""
    success, msg = memory_storage.test_connection()
    assert success is True
    assert "successful" in msg.lower()


def test_save_and_get_dataset(memory_storage):
    """Test dataset registration and lookup."""
    ds_id = memory_storage.save_dataset(
        name="user_activities.csv",
        row_count=9081,
        cadence="1286s",
        timestamp_col="created",
        metric_cols=["execution_time", "status_code"],
    )
    assert isinstance(ds_id, int)
    assert ds_id > 0

    record = memory_storage.get_dataset("user_activities.csv")
    assert record is not None
    assert record["id"] == ds_id
    assert record["name"] == "user_activities.csv"
    assert record["row_count"] == 9081
    assert record["cadence"] == "1286s"
    assert record["timestamp_col"] == "created"
    assert record["metric_cols"] == ["execution_time", "status_code"]

    # Test update existing dataset
    ds_id2 = memory_storage.save_dataset(
        name="user_activities.csv",
        row_count=9200,
        cadence="1286s",
        timestamp_col="created",
        metric_cols=["execution_time"],
    )
    assert ds_id2 == ds_id
    updated = memory_storage.get_dataset(ds_id)
    assert updated["row_count"] == 9200


def test_list_datasets(memory_storage):
    """Test listing all registered datasets."""
    memory_storage.save_dataset(name="dataset_a.csv", row_count=100)
    memory_storage.save_dataset(name="dataset_b.csv", row_count=200)

    datasets = memory_storage.list_datasets()
    assert len(datasets) == 2
    names = [d["name"] for d in datasets]
    assert "dataset_a.csv" in names
    assert "dataset_b.csv" in names


def test_save_run_with_detection_result(memory_storage):
    """Test persisting a DetectionResult with change events."""
    ds_id = memory_storage.save_dataset(name="api_metrics.csv", row_count=120)

    events = [
        ChangeEvent(
            index=33,
            timestamp=pd.Timestamp("2026-09-16 03:44:04"),
            direction="up",
            score=459.3,
            metadata={"shift": 272.07},
        ),
        ChangeEvent(
            index=66,
            timestamp=pd.Timestamp("2026-09-16 10:51:25"),
            direction="down",
            score=50.2,
            metadata={},
        ),
    ]

    result = DetectionResult(
        events=events,
        algorithm_name="ewma",
        params={"lambda_": 0.2, "control_limit": 3.0, "clip_z": 4.0},
        execution_time_ms=2.45,
    )

    run_id = memory_storage.save_run(
        dataset_name="api_metrics.csv",
        metric_name="execution_time_ms",
        detection_result=result,
        dataset_id=ds_id,
    )
    assert isinstance(run_id, int)
    assert run_id > 0

    run_data = memory_storage.get_run(run_id)
    assert run_data is not None
    assert run_data["dataset_name"] == "api_metrics.csv"
    assert run_data["metric_name"] == "execution_time_ms"
    assert run_data["algorithm_name"] == "ewma"
    assert run_data["total_events"] == 2
    assert run_data["params"]["lambda_"] == 0.2

    # Verify change events
    assert len(run_data["events"]) == 2
    ev0 = run_data["events"][0]
    assert ev0["observation_index"] == 33
    assert ev0["direction"] == "up"
    assert "2026-09-16" in ev0["event_timestamp"]
    assert ev0["score"] == 459.3
    assert ev0["details"]["shift"] == 272.07


def test_save_run_with_impact_report(memory_storage):
    """Test saving a detection run along with an impact report."""
    events = [ChangeEvent(index=40, timestamp=None, direction="up", score=12.5)]
    det_result = DetectionResult(events=events, algorithm_name="cusum")

    before_stats = SegmentStats(
        count=40,
        mean=50.0,
        std=5.0,
        median=49.8,
        min=40.0,
        max=60.0,
        p25=46.5,
        p75=53.5,
        iqr=7.0,
    )
    after_stats = SegmentStats(
        count=60,
        mean=150.0,
        std=8.0,
        median=149.5,
        min=135.0,
        max=165.0,
        p25=144.0,
        p75=155.0,
        iqr=11.0,
    )

    impact = MetricImpact(
        metric="latency",
        before_stats=before_stats,
        after_stats=after_stats,
        absolute_change=100.0,
        percent_change=200.0,
        median_change=99.7,
        std_ratio=1.6,
        cohen_d=14.3,
        effect_size_category="huge",
        t_statistic=72.5,
        p_value=1e-15,
        mann_whitney_p_value=1e-12,
        is_statistically_significant=True,
        severity="critical",
        summary_sentence="Latency increased by 100.00ms (200.0%) indicating a critical shift.",
    )

    run_id = memory_storage.save_run(
        dataset_name="latency_test.csv",
        metric_name="latency",
        detection_result=det_result,
        impact_report=impact,
    )

    run = memory_storage.get_run(run_id)
    assert run is not None
    assert len(run["impact_reports"]) == 1

    report = run["impact_reports"][0]
    assert report["severity"] == "critical"
    assert report["cohen_d"] == 14.3
    assert report["effect_size_category"] == "huge"
    assert report["pre_mean"] == 50.0
    assert report["post_mean"] == 150.0
    assert report["absolute_change"] == 100.0
    assert report["percent_change"] == 200.0
    assert report["pre_stats"]["count"] == 40
    assert report["post_stats"]["count"] == 60


def test_list_and_delete_runs(memory_storage):
    """Test listing runs with filters and cascading deletion."""
    r1 = DetectionResult(events=[], algorithm_name="sliding_window")
    r2 = DetectionResult(events=[], algorithm_name="ewma")

    id1 = memory_storage.save_run("service_a.csv", "cpu", r1)
    id2 = memory_storage.save_run("service_b.csv", "memory", r2)

    # Filter list
    all_runs = memory_storage.list_runs()
    assert len(all_runs) == 2

    filtered = memory_storage.list_runs(dataset_name="service_a.csv")
    assert len(filtered) == 1
    assert filtered[0]["id"] == id1

    # Delete run
    deleted = memory_storage.delete_run(id1)
    assert deleted is True
    assert memory_storage.get_run(id1) is None
    assert len(memory_storage.list_runs()) == 1


def test_delete_dataset_cascade(memory_storage):
    """Test that deleting a dataset cascades to its detection runs."""
    ds_id = memory_storage.save_dataset("cascade_test.csv", row_count=50)
    det = DetectionResult(events=[ChangeEvent(index=5, timestamp=None, direction="up", score=1.0)], algorithm_name="ewma")
    run_id = memory_storage.save_run("cascade_test.csv", "val", det, dataset_id=ds_id)

    assert memory_storage.get_run(run_id) is not None
    memory_storage.delete_dataset(ds_id)

    assert memory_storage.get_dataset(ds_id) is None
    assert memory_storage.get_run(run_id) is None
