"""CI/CD Performance Regression Quality Gate for PatternTime.

Evaluates candidate benchmarks and telemetry against baselines or SLAs,
detects statistical regressions (Cohen's d, percent change, error rates),
and produces exit codes, terminal output, and GitHub Step Summary markdown.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from patterntime.detectors import (
    CusumDetector,
    DetectionResult,
    EWMADetector,
    RobustCusumDetector,
    SlidingWindowDetector,
)
from patterntime.impact import analyze_metric_shift
from patterntime.profiler import profile_endpoints
from patterntime.validator import (
    detect_metric_columns,
    detect_timestamp_column,
)


@dataclass
class GateRuleConfig:
    """Configurable thresholds and rules for the CI/CD performance gate."""

    metric: str | None = None
    baseline_path: Path | str | None = None
    max_cohen_d: float = 0.8
    max_pct_increase: float = 20.0
    max_error_rate_pct: float = 5.0
    fail_on_severity: list[str] = field(default_factory=lambda: ["critical", "high"])
    per_endpoint: bool = False
    group_col: str | None = None
    status_col: str | None = None
    algorithm: str = "ewma"
    threshold: float | None = None
    min_samples: int = 15


@dataclass
class GateCheckResult:
    """The outcome of a single gate check / rule assertion."""

    name: str
    passed: bool
    severity: str  # "pass", "warning", "failure"
    metric: str
    observed_value: float | str
    threshold_value: float | str
    comparator: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "metric": self.metric,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "comparator": self.comparator,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class GateReport:
    """Aggregated CI/CD quality gate evaluation report."""

    dataset_name: str
    baseline_name: str | None
    passed: bool
    total_checks: int
    failed_checks: int
    warning_checks: int
    checks: list[GateCheckResult]
    regressed_endpoints: list[str]
    summary_text: str
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert GateReport to JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "baseline_name": self.baseline_name,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "checks": [c.to_dict() for c in self.checks],
            "regressed_endpoints": self.regressed_endpoints,
            "summary_text": self.summary_text,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }

    def to_terminal_text(self, use_color: bool = True) -> str:
        """Render a formatted terminal banner and report."""
        # ANSI colors
        green = "\033[92m" if use_color else ""
        red = "\033[91m" if use_color else ""
        yellow = "\033[93m" if use_color else ""
        cyan = "\033[96m" if use_color else ""
        bold = "\033[1m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        status_tag = f"{green}{bold}[PASSED]{reset}" if self.passed else f"{red}{bold}[FAILED]{reset}"
        lines = [
            f"{bold}{'=' * 78}{reset}",
            f"🚦 {bold}PATTERNTIME PERFORMANCE GATE: {status_tag}",
            f"{bold}{'=' * 78}{reset}",
            f"  Target Telemetry:   {cyan}{self.dataset_name}{reset}",
        ]
        if self.baseline_name:
            lines.append(f"  Baseline Baseline:  {cyan}{self.baseline_name}{reset}")

        lines.extend([
            f"  Total Checks:       {self.total_checks}",
            f"  Failed Checks:      {red if self.failed_checks else green}{self.failed_checks}{reset}",
            f"  Evaluation Time:    {self.execution_time_ms:.2f} ms",
            f"  Gate Decision:      {green + 'ALLOW DEPLOYMENT ✓' if self.passed else red + 'BLOCK DEPLOYMENT ✖'}{reset}",
            "",
            f"{bold}Detailed Assertion Results:{reset}",
        ])

        for c in self.checks:
            icon = f"{green}✓ PASS{reset}" if c.passed else f"{red}✖ FAIL{reset}"
            lines.append(f"  {icon} [{c.metric}] {c.name}")
            lines.append(f"         {c.message}")
            if not c.passed and "effect_size" in c.details:
                lines.append(f"         Effect size: {c.details.get('effect_size')} (Cohen's d={c.details.get('cohen_d')})")

        if not self.passed:
            lines.extend([
                "",
                f"{red}{bold}Action Required:{reset}",
                f"  {red}Performance degradation exceeds defined SLA or statistical tolerances.{reset}",
                f"  {red}Inspect the regressions listed above before promoting this build.{reset}",
            ])
        else:
            lines.extend([
                "",
                f"{green}✓ All performance assertions met. No statistically significant regressions detected.{reset}",
            ])

        lines.append(f"{bold}{'=' * 78}{reset}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render a GitHub Actions Step Summary markdown string."""
        badge = "🟢 **PASSED**" if self.passed else "🔴 **FAILED**"
        lines = [
            "## 🚦 PatternTime Performance Quality Gate",
            "",
            f"**Decision:** {badge} — `{self.dataset_name}`",
        ]
        if self.baseline_name:
            lines.append(f"- **Baseline:** `{self.baseline_name}`")
        lines.extend([
            f"- **Checks Evaluated:** {self.total_checks} | **Failed:** {self.failed_checks} | **Warnings:** {self.warning_checks}",
            f"- **Execution Time:** {self.execution_time_ms:.2f} ms",
            "",
            "### Summary",
            f"> {self.summary_text}",
            "",
            "### Assertions Table",
            "| Status | Check / Route | Metric | Observed | Threshold | Details |",
            "|:---:|:---|:---|:---:|:---:|:---|",
        ])

        for c in self.checks:
            icon = "✅ Pass" if c.passed else "❌ **Fail**"
            obs = str(c.observed_value)
            thresh = f"{c.comparator} {c.threshold_value}"
            lines.append(f"| {icon} | `{c.name}` | `{c.metric}` | {obs} | {thresh} | {c.message} |")

        if self.regressed_endpoints:
            lines.extend([
                "",
                "### 🚨 Regressed Endpoints Requiring Remediation",
                "The following routes experienced statistically significant latency or error regressions:",
            ])
            for ep in self.regressed_endpoints:
                lines.append(f"- ❌ `{ep}`")

        lines.extend([
            "",
            "---",
            "*Report generated automatically by [PatternTime](https://github.com/falgunisardhara/patterntime)*",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Evaluation Logic
# ---------------------------------------------------------------------------
def evaluate_gate(
    df: pd.DataFrame,
    dataset_name: str,
    config: GateRuleConfig,
    df_baseline: pd.DataFrame | None = None,
    baseline_name: str | None = None,
) -> GateReport:
    """
    Run CI/CD performance regression gate evaluation.

    Args:
        df: Candidate telemetry DataFrame.
        dataset_name: Name or path of the candidate dataset.
        config: GateRuleConfig instance specifying thresholds.
        df_baseline: Optional baseline DataFrame to compare against.
        baseline_name: Optional baseline dataset name.

    Returns:
        GateReport containing all assertion results and pass/fail decision.
    """
    start_time = time.perf_counter()
    checks: list[GateCheckResult] = []
    regressed_endpoints: list[str] = []

    # Detect metric column
    ts_col = detect_timestamp_column(df)
    metric_cols = detect_metric_columns(df, timestamp_col=ts_col)

    target_metric = config.metric
    if not target_metric:
        if metric_cols:
            target_metric = metric_cols[0]
        else:
            # Fallback numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                target_metric = num_cols[0]
            else:
                raise ValueError("No numeric metric column found in dataset")

    if target_metric not in df.columns:
        raise ValueError(f"Specified metric '{target_metric}' not found in candidate dataset")

    # -----------------------------------------------------------------------
    # 1. Mode: Baseline Comparison (Candidate vs Baseline)
    # -----------------------------------------------------------------------
    if df_baseline is not None:
        if target_metric not in df_baseline.columns:
            raise ValueError(f"Specified metric '{target_metric}' not found in baseline dataset")

        series_base = df_baseline[target_metric].dropna()
        series_cand = df[target_metric].dropna()

        impact = analyze_metric_shift(series_base, series_cand, metric_name=target_metric)

        # Check A: Statistical Effect Size (Cohen's d)
        # Note: only fail if latency INCREASED (d > 0)
        cohen_d_val = round(float(impact.cohen_d), 3)
        pct_change_val = round(float(impact.percent_change or 0.0), 2)
        p_val = impact.p_value

        is_stat_sig = p_val is not None and p_val < 0.05
        is_regressed_d = (cohen_d_val > config.max_cohen_d) and is_stat_sig
        is_regressed_pct = pct_change_val > config.max_pct_increase

        d_passed = not (is_regressed_d and pct_change_val > 0)
        checks.append(
            GateCheckResult(
                name="Baseline Effect Size (Cohen's d)",
                passed=d_passed,
                severity="pass" if d_passed else "failure",
                metric=target_metric,
                observed_value=f"d={cohen_d_val:.2f}",
                threshold_value=f"d<={config.max_cohen_d}",
                comparator="<=",
                message=(
                    f"Effect size is {impact.effect_size_category} (d={cohen_d_val:.2f}, p={p_val})"
                    if d_passed
                    else f"Statistically significant degradation: d={cohen_d_val:.2f} exceeds threshold {config.max_cohen_d} (p={p_val})"
                ),
                details={
                    "cohen_d": cohen_d_val,
                    "effect_size": impact.effect_size_category,
                    "p_value": p_val,
                },
            )
        )

        # Check B: Percent Increase
        pct_passed = not is_regressed_pct
        checks.append(
            GateCheckResult(
                name="Baseline Percent Change",
                passed=pct_passed,
                severity="pass" if pct_passed else "failure",
                metric=target_metric,
                observed_value=f"{pct_change_val:+.1f}%",
                threshold_value=f"<={config.max_pct_increase:.1f}%",
                comparator="<=",
                message=(
                    f"Mean shifted by {pct_change_val:+.1f}% (from {impact.before_stats.mean:.1f} to {impact.after_stats.mean:.1f})"
                    if pct_passed
                    else f"Latency increase of {pct_change_val:+.1f}% exceeds max allowed increase of {config.max_pct_increase:.1f}%"
                ),
                details={
                    "percent_change": pct_change_val,
                    "before_mean": impact.before_stats.mean,
                    "after_mean": impact.after_stats.mean,
                },
            )
        )

    # -----------------------------------------------------------------------
    # 2. Mode: In-Series Change Point Detection (Single candidate dataset)
    # -----------------------------------------------------------------------
    else:
        # Instantiate detector
        detector: Any
        if config.algorithm == "sliding_window":
            win = 12
            thresh = config.threshold if config.threshold is not None else 50.0
            detector = SlidingWindowDetector(window_size=win, threshold=thresh)
        elif config.algorithm == "cusum":
            thresh = config.threshold if config.threshold is not None else 10.0
            detector = CusumDetector(threshold=thresh)
        elif config.algorithm == "robust_cusum":
            thresh = config.threshold if config.threshold is not None else 10.0
            detector = RobustCusumDetector(threshold=thresh)
        else:  # ewma
            ctrl = config.threshold if config.threshold is not None else 3.0
            detector = EWMADetector(control_limit=ctrl)

        series = df[target_metric].dropna()
        ts_series = df[ts_col] if ts_col and ts_col in df.columns else None
        det_result: DetectionResult = detector.detect(series=series, timestamps=ts_series)

        if not det_result.events:
            checks.append(
                GateCheckResult(
                    name="In-Series Change Point Detection",
                    passed=True,
                    severity="pass",
                    metric=target_metric,
                    observed_value="0 shifts",
                    threshold_value="No critical shifts",
                    comparator="==",
                    message="Time series remained in steady regime throughout execution.",
                )
            )
        else:
            # Evaluate each detected event's impact
            has_critical_shift = False
            for ev in det_result.events:
                # Segment before and after event
                idx = ev.index
                before = series.iloc[max(0, idx - 30) : idx]
                after = series.iloc[idx : min(len(series), idx + 30)]

                if len(before) >= 5 and len(after) >= 5:
                    ev_impact = analyze_metric_shift(before, after, metric_name=target_metric)
                    is_bad_severity = ev_impact.severity in config.fail_on_severity
                    is_bad_cohen = (ev_impact.cohen_d > config.max_cohen_d) and (ev_impact.p_value or 1.0) < 0.05
                    is_increase = (ev_impact.percent_change or 0.0) > 0

                    if is_bad_severity and is_bad_cohen and is_increase:
                        has_critical_shift = True
                        checks.append(
                            GateCheckResult(
                                name=f"In-Series Shift @ observation #{idx}",
                                passed=False,
                                severity="failure",
                                metric=target_metric,
                                observed_value=f"d={ev_impact.cohen_d:.2f} ({ev_impact.severity.upper()})",
                                threshold_value=f"d<={config.max_cohen_d}",
                                comparator="<=",
                                message=(
                                    f"Detected change point #{idx} has {ev_impact.severity.upper()} severity: "
                                    f"shift {ev_impact.absolute_change:+.1f} ({ev_impact.percent_change:+.1f}%), Cohen's d={ev_impact.cohen_d:.2f}"
                                ),
                                details={"index": idx, "cohen_d": ev_impact.cohen_d, "severity": ev_impact.severity},
                            )
                        )

            if not has_critical_shift:
                checks.append(
                    GateCheckResult(
                        name="In-Series Change Point Detection",
                        passed=True,
                        severity="pass",
                        metric=target_metric,
                        observed_value=f"{len(det_result.events)} benign shifts",
                        threshold_value="No critical shifts",
                        comparator="==",
                        message=f"{len(det_result.events)} minor fluctuations detected within acceptable SLA tolerance.",
                    )
                )

    # -----------------------------------------------------------------------
    # 3. Mode: Per-Endpoint Route Profiling Gate (--per-endpoint)
    # -----------------------------------------------------------------------
    if config.per_endpoint:
        profile_report = profile_endpoints(
            df=df,
            group_col=config.group_col,
            metric_col=target_metric,
            status_col=config.status_col,
            algorithm=config.algorithm,
            min_samples=config.min_samples,
            dataset_name=dataset_name,
        )

        for ep in profile_report.endpoints:
            if ep.request_count < config.min_samples:
                continue

            # Check A: Error rate SLA
            if ep.error_rate_pct > config.max_error_rate_pct:
                regressed_endpoints.append(ep.endpoint_name)
                checks.append(
                    GateCheckResult(
                        name=f"Route Error Rate: {ep.endpoint_name}",
                        passed=False,
                        severity="failure",
                        metric="error_rate",
                        observed_value=f"{ep.error_rate_pct:.1f}%",
                        threshold_value=f"<={config.max_error_rate_pct:.1f}%",
                        comparator="<=",
                        message=f"Route '{ep.endpoint_name}' error rate of {ep.error_rate_pct:.1f}% exceeds max {config.max_error_rate_pct:.1f}%",
                        details={"endpoint": ep.endpoint_name, "error_rate_pct": ep.error_rate_pct},
                    )
                )

            # Check B: Regression severity & Cohen's d
            if ep.max_severity in config.fail_on_severity and ep.max_cohen_d > config.max_cohen_d:
                if ep.endpoint_name not in regressed_endpoints:
                    regressed_endpoints.append(ep.endpoint_name)
                checks.append(
                    GateCheckResult(
                        name=f"Route Latency Regression: {ep.endpoint_name}",
                        passed=False,
                        severity="failure",
                        metric=target_metric,
                        observed_value=f"d={ep.max_cohen_d:.2f} ({ep.max_severity.upper()})",
                        threshold_value=f"d<={config.max_cohen_d}",
                        comparator="<=",
                        message=f"Route '{ep.endpoint_name}' suffered {ep.max_severity.upper()} regression (d={ep.max_cohen_d:.2f}, p95={ep.p95_latency:.1f}ms)",
                        details={"endpoint": ep.endpoint_name, "cohen_d": ep.max_cohen_d, "severity": ep.max_severity},
                    )
                )

        if not regressed_endpoints:
            checks.append(
                GateCheckResult(
                    name="Per-Endpoint Regression Isolation",
                    passed=True,
                    severity="pass",
                    metric=target_metric,
                    observed_value=f"{len(profile_report.endpoints)} routes healthy",
                    threshold_value="0 route regressions",
                    comparator="==",
                    message=f"All {len(profile_report.endpoints)} profiled endpoints conform to latency and error rate SLAs.",
                )
            )

    # -----------------------------------------------------------------------
    # Aggregate Report
    # -----------------------------------------------------------------------
    exec_ms = (time.perf_counter() - start_time) * 1000.0
    failed_count = sum(1 for c in checks if not c.passed)
    warning_count = sum(1 for c in checks if c.severity == "warning")
    gate_passed = failed_count == 0

    if gate_passed:
        summary_text = f"All {len(checks)} performance checks passed successfully. Build meets SLA criteria."
    else:
        summary_text = f"Quality gate failed with {failed_count} failing assertion(s) across telemetry."

    return GateReport(
        dataset_name=dataset_name,
        baseline_name=baseline_name,
        passed=gate_passed,
        total_checks=len(checks),
        failed_checks=failed_count,
        warning_checks=warning_count,
        checks=checks,
        regressed_endpoints=regressed_endpoints,
        summary_text=summary_text,
        execution_time_ms=exec_ms,
    )
