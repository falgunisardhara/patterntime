"""
Unit tests for the patterntime CLI application.
"""

import json
from pathlib import Path
import pytest

from patterntime.cli import main


def test_cli_help(capsys):
    """Test CLI top-level help output."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "PatternTime: Interactive AI/ML Time-Series Investigation Laboratory" in captured.out


def test_cli_db_commands(capsys, monkeypatch, tmp_path):
    """Test db subcommands with memory sqlite database."""
    test_db = f"sqlite:///{tmp_path / 'cli_test.db'}"
    monkeypatch.setenv("DATABASE_URL", test_db)

    # Test init
    code = main(["db", "init"])
    assert code == 0
    captured = capsys.readouterr()
    assert "verified / created" in captured.out

    # Test test
    code = main(["db", "test"])
    assert code == 0
    captured = capsys.readouterr()
    assert "[OK]" in captured.out

    # Test status
    code = main(["db", "status"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Connected:      YES" in captured.out
    assert "change_events" in captured.out


def test_cli_validate(capsys):
    """Test validate command on sample data."""
    csv_path = "data/sample_api_metrics.csv"
    code = main(["validate", csv_path])
    assert code == 0
    captured = capsys.readouterr()
    assert "Status:            VALID" in captured.out
    assert "response_time_ms" in captured.out


def test_cli_validate_json(capsys):
    """Test validate command JSON output."""
    csv_path = "data/sample_api_metrics.csv"
    code = main(["validate", csv_path, "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["valid"] is True
    assert data["total_rows"] == 120


def test_cli_detect_and_runs(capsys, monkeypatch, tmp_path):
    """Test detect command with database persistence and run query."""
    test_db = f"sqlite:///{tmp_path / 'runs_test.db'}"
    monkeypatch.setenv("DATABASE_URL", test_db)

    # Initialize DB
    main(["db", "init"])
    capsys.readouterr()

    # Run detection
    code = main([
        "detect",
        "data/sample_api_metrics.csv",
        "--metric", "response_time_ms",
        "--algorithm", "sliding_window",
        "--threshold", "1.5",
        "--save",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "Detection Results: sample_api_metrics.csv" in captured.out
    assert "Saved to MySQL Run: #1" in captured.out or "Run: #1" in captured.out

    # List runs
    code = main(["runs", "list"])
    assert code == 0
    captured = capsys.readouterr()
    assert "sample_api_metrics.csv" in captured.out

    # Show run
    code = main(["runs", "show", "1"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Detection Run #1" in captured.out
    assert "response_time_ms" in captured.out

    # List datasets
    code = main(["datasets", "list"])
    assert code == 0
    captured = capsys.readouterr()
    assert "sample_api_metrics.csv" in captured.out
