"""
Unit tests for the PatternTime HTTP and API server.
"""

import json
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen
import pytest

from patterntime.server import PatternTimeHandler
from patterntime.storage import DatabaseStorage


@pytest.fixture
def test_server(monkeypatch, tmp_path):
    """Start an ephemeral test server running against an isolated SQLite DB."""
    test_db = f"sqlite:///{tmp_path / 'server_test.db'}"
    monkeypatch.setenv("DATABASE_URL", test_db)

    # Initialize tables
    storage = DatabaseStorage(auto_create_tables=True)
    storage.save_dataset("api_test.csv", row_count=100)

    server = HTTPServer(("127.0.0.1", 0), PatternTimeHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url

    server.shutdown()
    server.server_close()


def test_api_db_status(test_server):
    """Test GET /api/db/status endpoint."""
    with urlopen(f"{test_server}/api/db/status") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["connected"] is True
        assert "change_events" in data["tables"]


def test_api_datasets(test_server):
    """Test GET /api/datasets endpoint."""
    with urlopen(f"{test_server}/api/datasets") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "datasets" in data
        assert len(data["datasets"]) == 1
        assert data["datasets"][0]["name"] == "api_test.csv"


def test_api_runs_lifecycle(test_server):
    """Test POST /api/runs and GET /api/runs endpoints."""
    payload = {
        "dataset_name": "api_test.csv",
        "metric_name": "latency",
        "algorithm_name": "sliding_window",
        "params": {"window_size": 25},
        "execution_time_ms": 3.12,
        "events": [
            {"index": 15, "direction": "up", "score": 10.5, "metadata": {"diff": 5.2}}
        ],
    }

    req = Request(
        f"{test_server}/api/runs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        assert resp.status == 201
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "success"
        run_id = res["run_id"]
        assert run_id > 0

    # Retrieve run list
    with urlopen(f"{test_server}/api/runs") as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["runs"]) == 1
        assert data["runs"][0]["id"] == run_id

    # Retrieve individual run
    with urlopen(f"{test_server}/api/runs/{run_id}") as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert data["run"]["id"] == run_id
        assert len(data["run"]["events"]) == 1

    # Delete run
    del_req = Request(f"{test_server}/api/runs/{run_id}", method="DELETE")
    with urlopen(del_req) as resp:
        assert resp.status == 200
        del_data = json.loads(resp.read().decode("utf-8"))
        assert del_data["status"] == "success"

    # Verify deleted
    with urlopen(f"{test_server}/api/runs") as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["runs"]) == 0


def test_api_profile_endpoint(test_server):
    """Test GET /api/profile endpoint on user_activities.csv."""
    with urlopen(f"{test_server}/api/profile?dataset=user_activities.csv&min_samples=50") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "endpoints" in data
        assert "regressed_endpoints" in data
        assert data["total_requests"] == 9081
        assert len(data["endpoints"]) > 5


def test_api_telemetry_lifecycle(test_server):
    """Test POST and GET /api/telemetry endpoints."""
    from urllib.error import HTTPError
    from pathlib import Path

    test_ds = "tmp_telemetry_api_test.csv"
    cleanup_path = Path("data") / test_ds

    try:
        # 1. Post a batch of telemetry records
        payload = {
            "dataset": test_ds,
            "service": "billing-api",
            "records": [
                {"timestamp": "2026-11-01 10:00:00", "route": "/charge", "duration_ms": 140.5, "status": 200},
                {"timestamp": "2026-11-01 10:00:05", "route": "/charge", "duration_ms": 145.2, "status": 200},
                {"timestamp": "2026-11-01 10:00:10", "route": "/refund", "duration_ms": 210.0, "status": 200},
            ],
        }

        req = Request(
            f"{test_server}/api/telemetry",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            assert resp.status == 201
            res = json.loads(resp.read().decode("utf-8"))
            assert res["status"] == "success"
            assert res["dataset"] == test_ds
            assert res["records_ingested"] == 3
            assert res["total_rows"] == 3
            assert "duration_ms" in res["metrics"]

        # 2. Query telemetry slice
        with urlopen(f"{test_server}/api/telemetry?dataset={test_ds}&limit=2") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["dataset"] == test_ds
            assert data["total_rows"] == 3
            assert len(data["records"]) == 2

        # 3. Query telemetry with route filter
        with urlopen(f"{test_server}/api/telemetry?dataset={test_ds}&route=/refund") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["filtered_rows"] == 1
            assert data["records"][0]["route"] == "/refund"

        # 4. Enumerate telemetry datasets
        with urlopen(f"{test_server}/api/telemetry") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "datasets" in data
            names = [d["name"] for d in data["datasets"]]
            assert test_ds in names

        # 5. Bad request on empty records
        bad_req = Request(
            f"{test_server}/api/telemetry",
            data=json.dumps({"records": []}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(bad_req)
        assert exc_info.value.code == 400

    finally:
        if cleanup_path.exists():
            cleanup_path.unlink()

