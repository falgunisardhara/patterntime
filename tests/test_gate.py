"""Unit tests for CI/CD Performance Regression Quality Gate (gate.py and CLI gate)."""

from pathlib import Path

import pandas as pd
import pytest

from patterntime.cli import main
from patterntime.gate import (
    GateCheckResult,
    GateReport,
    GateRuleConfig,
    evaluate_gate,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Load baseline clean metrics DataFrame."""
    p = Path("data/no_change_metrics.csv")
    return pd.read_csv(p)


@pytest.fixture
def shift_df() -> pd.DataFrame:
    """Load shifted metrics DataFrame."""
    p = Path("data/change_size_150ms.csv")
    return pd.read_csv(p)


@pytest.fixture
def user_activities_df() -> pd.DataFrame:
    """Load CakePHP user activities DataFrame."""
    p = Path("data/user_activities.csv")
    return pd.read_csv(p)


def test_gate_clean_dataset_passes(clean_df: pd.DataFrame) -> None:
    """Verify that a clean dataset without regressions passes the quality gate."""
    config = GateRuleConfig(metric="response_time_ms")
    report = evaluate_gate(df=clean_df, dataset_name="no_change_metrics.csv", config=config)

    assert report.passed is True
    assert report.failed_checks == 0
    assert report.total_checks >= 1
    assert "no statistically significant regressions" in report.to_terminal_text(use_color=False).lower()


def test_gate_shift_dataset_fails(shift_df: pd.DataFrame) -> None:
    """Verify that a dataset containing an in-series shift fails the quality gate."""
    config = GateRuleConfig(metric="response_time_ms", max_cohen_d=0.8)
    report = evaluate_gate(df=shift_df, dataset_name="change_size_150ms.csv", config=config)

    assert report.passed is False
    assert report.failed_checks >= 1
    assert "BLOCK DEPLOYMENT" in report.to_terminal_text(use_color=False)


def test_gate_baseline_comparison_failure(clean_df: pd.DataFrame, shift_df: pd.DataFrame) -> None:
    """Verify that comparing a regressed candidate to a baseline fails the gate."""
    config = GateRuleConfig(
        metric="response_time_ms",
        max_cohen_d=0.8,
        max_pct_increase=15.0,
    )
    report = evaluate_gate(
        df=shift_df,
        dataset_name="candidate.csv",
        config=config,
        df_baseline=clean_df,
        baseline_name="baseline.csv",
    )

    assert report.passed is False
    assert report.failed_checks >= 1
    assert report.baseline_name == "baseline.csv"


def test_gate_baseline_comparison_passing(clean_df: pd.DataFrame) -> None:
    """Verify that comparing identical or non-shifted candidate to baseline passes."""
    config = GateRuleConfig(
        metric="response_time_ms",
        max_cohen_d=0.8,
        max_pct_increase=20.0,
    )
    report = evaluate_gate(
        df=clean_df,
        dataset_name="candidate.csv",
        config=config,
        df_baseline=clean_df,
        baseline_name="baseline.csv",
    )

    assert report.passed is True
    assert report.failed_checks == 0


def test_gate_per_endpoint_detects_regression(user_activities_df: pd.DataFrame) -> None:
    """Verify that --per-endpoint isolates route regressions on real telemetry."""
    config = GateRuleConfig(
        metric="execution_time_ms",
        per_endpoint=True,
        min_samples=30,
        max_cohen_d=0.8,
    )
    report = evaluate_gate(
        df=user_activities_df,
        dataset_name="user_activities.csv",
        config=config,
    )

    assert report.passed is False
    assert "api/v1/resend-otp" in report.regressed_endpoints


def test_gate_markdown_and_dict_serialization(shift_df: pd.DataFrame) -> None:
    """Verify Markdown generation and dictionary serialization."""
    config = GateRuleConfig(metric="response_time_ms")
    report = evaluate_gate(df=shift_df, dataset_name="change_size_150ms.csv", config=config)

    md = report.to_markdown()
    assert "## 🚦 PatternTime Performance Quality Gate" in md
    assert "| Status | Check / Route |" in md

    d = report.to_dict()
    assert d["dataset_name"] == "change_size_150ms.csv"
    assert isinstance(d["checks"], list)
    assert len(d["checks"]) > 0


def test_cli_gate_exit_codes(tmp_path: Path) -> None:
    """Verify CLI exit codes: 0 for pass, 1 for fail, 2 for missing file."""
    # Clean dataset -> exit code 0
    exit_pass = main(["gate", "data/no_change_metrics.csv", "--quiet"])
    assert exit_pass == 0

    # Shifted dataset -> exit code 1
    exit_fail = main(["gate", "data/change_size_150ms.csv", "--quiet"])
    assert exit_fail == 1

    # Non-existent file -> exit code 2
    exit_err = main(["gate", "data/does_not_exist_file.csv", "--quiet"])
    assert exit_err == 2


def test_cli_gate_output_summary(tmp_path: Path) -> None:
    """Verify CLI writes GitHub Actions Step Summary markdown to designated path."""
    summary_file = tmp_path / "ci_summary.md"
    exit_code = main([
        "gate",
        "data/change_size_150ms.csv",
        "--output-summary",
        str(summary_file),
        "--format",
        "json",
    ])
    assert exit_code == 1
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "PatternTime Performance Quality Gate" in content
    assert "change_size_150ms.csv" in content
