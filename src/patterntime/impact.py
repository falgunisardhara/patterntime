from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from patterntime.validator import detect_metric_columns, detect_timestamp_column


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class SegmentStats:
    """Summary statistics for a time-series segment."""

    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    p25: float
    p75: float
    iqr: float

    def to_dict(self) -> dict[str, Any]:
        """Convert SegmentStats to a JSON-serializable dictionary."""
        return {
            "count": self.count,
            "mean": round(float(self.mean), 3),
            "std": round(float(self.std), 3),
            "median": round(float(self.median), 3),
            "min": round(float(self.min), 3),
            "max": round(float(self.max), 3),
            "p25": round(float(self.p25), 3),
            "p75": round(float(self.p75), 3),
            "iqr": round(float(self.iqr), 3),
        }


@dataclass
class MetricImpact:
    """Detailed statistical comparison before vs after a detected change point."""

    metric: str
    before_stats: SegmentStats
    after_stats: SegmentStats
    absolute_change: float
    percent_change: float | None
    median_change: float
    std_ratio: float | None
    cohen_d: float
    effect_size_category: str  # "negligible", "small", "medium", "large", "huge"
    t_statistic: float | None
    p_value: float | None
    mann_whitney_p_value: float | None
    is_statistically_significant: bool  # True if p < 0.05
    severity: str  # "negligible", "low", "medium", "high", "critical"
    summary_sentence: str

    def to_dict(self) -> dict[str, Any]:
        """Convert MetricImpact to a JSON-serializable dictionary."""
        return {
            "metric": self.metric,
            "before": self.before_stats.to_dict(),
            "after": self.after_stats.to_dict(),
            "absolute_change": round(float(self.absolute_change), 3),
            "percent_change": (
                round(float(self.percent_change), 2)
                if self.percent_change is not None
                else None
            ),
            "median_change": round(float(self.median_change), 3),
            "std_ratio": (
                round(float(self.std_ratio), 3) if self.std_ratio is not None else None
            ),
            "cohen_d": round(float(self.cohen_d), 3),
            "effect_size_category": self.effect_size_category,
            "t_statistic": (
                round(float(self.t_statistic), 3) if self.t_statistic is not None else None
            ),
            "p_value": float(f"{self.p_value:.3e}") if self.p_value is not None else None,
            "mann_whitney_p_value": (
                float(f"{self.mann_whitney_p_value:.3e}")
                if self.mann_whitney_p_value is not None
                else None
            ),
            "is_statistically_significant": self.is_statistically_significant,
            "severity": self.severity,
            "summary_sentence": self.summary_sentence,
        }


@dataclass
class ChangeImpactReport:
    """Comprehensive impact report across all metrics for a given change event."""

    change_index: int
    timestamp: pd.Timestamp | None
    window_size: int | None
    metrics: dict[str, MetricImpact]
    overall_severity: str

    def to_dict(self) -> dict[str, Any]:
        """Convert ChangeImpactReport to a JSON-serializable dictionary."""
        return {
            "change_index": self.change_index,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "window_size": self.window_size,
            "overall_severity": self.overall_severity,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


# ---------------------------------------------------------------------------
# Statistical Helper Functions
# ---------------------------------------------------------------------------
def compute_segment_stats(series: pd.Series) -> SegmentStats:
    """Compute comprehensive descriptive statistics for a time-series segment."""
    clean = series.dropna()
    count = len(clean)

    if count == 0:
        return SegmentStats(
            count=0,
            mean=0.0,
            std=0.0,
            median=0.0,
            min=0.0,
            max=0.0,
            p25=0.0,
            p75=0.0,
            iqr=0.0,
        )

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if count > 1 else 0.0
    median_val = float(clean.median())
    min_val = float(clean.min())
    max_val = float(clean.max())
    p25_val = float(clean.quantile(0.25))
    p75_val = float(clean.quantile(0.75))
    iqr_val = p75_val - p25_val

    return SegmentStats(
        count=count,
        mean=mean_val,
        std=0.0 if math.isnan(std_val) else std_val,
        median=median_val,
        min=min_val,
        max=max_val,
        p25=p25_val,
        p75=p75_val,
        iqr=iqr_val,
    )


def calculate_cohen_d(before: pd.Series, after: pd.Series) -> tuple[float, str]:
    """
    Compute Cohen's d effect size for two independent samples.
    Returns (effect_size, category).
    """
    n1, n2 = len(before), len(after)
    if n1 < 2 or n2 < 2:
        return 0.0, "negligible"

    s1 = float(before.std(ddof=1))
    s2 = float(after.std(ddof=1))

    # Pooled standard deviation
    numerator = (n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)
    denominator = n1 + n2 - 2

    if denominator <= 0:
        return 0.0, "negligible"

    s_pooled = math.sqrt(numerator / denominator)
    mean_diff = float(after.mean() - before.mean())

    if s_pooled == 0.0 or math.isclose(s_pooled, 0.0):
        if math.isclose(mean_diff, 0.0):
            return 0.0, "negligible"
        d = 99.0 if mean_diff > 0 else -99.0
    else:
        d = mean_diff / s_pooled

    abs_d = abs(d)
    if abs_d < 0.2:
        category = "negligible"
    elif abs_d < 0.5:
        category = "small"
    elif abs_d < 0.8:
        category = "medium"
    elif abs_d < 2.0:
        category = "large"
    else:
        category = "huge"

    return d, category


def classify_severity(
    percent_change: float | None,
    cohen_d: float,
    p_value: float | None,
) -> str:
    """
    Classify the severity of a metric shift into one of:
    'negligible', 'low', 'medium', 'high', 'critical'
    """
    # If not statistically significant, classify as negligible
    if p_value is not None and p_value >= 0.05:
        return "negligible"

    abs_d = abs(cohen_d)
    abs_pct = abs(percent_change) if percent_change is not None else 0.0

    if abs_d < 0.2 and abs_pct < 5.0:
        return "negligible"
    if abs_d < 0.5 and abs_pct < 15.0:
        return "low"
    if abs_d < 1.0 or abs_pct < 40.0:
        return "medium"
    if abs_d < 2.5 or abs_pct < 100.0:
        return "high"

    return "critical"


SEVERITY_RANK = {
    "negligible": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# ---------------------------------------------------------------------------
# Core Analyzer Functions
# ---------------------------------------------------------------------------
def analyze_metric_shift(
    before_series: pd.Series,
    after_series: pd.Series,
    metric_name: str,
) -> MetricImpact:
    """
    Analyze the before vs after shift for a single metric.
    """
    before_clean = before_series.dropna()
    after_clean = after_series.dropna()

    before_stats = compute_segment_stats(before_clean)
    after_stats = compute_segment_stats(after_clean)

    abs_change = after_stats.mean - before_stats.mean
    median_change = after_stats.median - before_stats.median

    percent_change: float | None = None
    if not math.isclose(before_stats.mean, 0.0):
        percent_change = (abs_change / abs(before_stats.mean)) * 100.0

    std_ratio: float | None = None
    if not math.isclose(before_stats.std, 0.0):
        std_ratio = after_stats.std / before_stats.std

    cohen_d, effect_category = calculate_cohen_d(before_clean, after_clean)

    # Hypothesis tests (Welch's t-test and Mann-Whitney U test)
    t_stat: float | None = None
    p_val: float | None = None
    mw_p: float | None = None

    if len(before_clean) >= 2 and len(after_clean) >= 2:
        try:
            # Check for zero variance
            if before_stats.std > 0 or after_stats.std > 0:
                res = stats.ttest_ind(after_clean, before_clean, equal_var=False)
                t_stat = float(res.statistic)
                p_val = float(res.pvalue)

                # Non-parametric Mann-Whitney U test
                mw_res = stats.mannwhitneyu(after_clean, before_clean, alternative="two-sided")
                mw_p = float(mw_res.pvalue)
            else:
                # Both zero variance
                p_val = 0.0 if not math.isclose(abs_change, 0.0) else 1.0
                mw_p = p_val
        except Exception:
            p_val = None
            t_stat = None
            mw_p = None

    is_sig = bool(p_val is not None and p_val < 0.05)
    severity = classify_severity(percent_change, cohen_d, p_val)

    # Build human-readable narrative summary
    direction = "increased" if abs_change > 0 else "decreased"
    pct_str = f" ({percent_change:+.1f}%)" if percent_change is not None else ""
    pval_str = f", p={p_val:.2e}" if p_val is not None else ""
    summary = (
        f"{metric_name} {direction} by {abs(abs_change):.2f}{pct_str} "
        f"[effect: {effect_category} (d={cohen_d:.2f}){pval_str}] "
        f"-> Severity: {severity.upper()}"
    )

    return MetricImpact(
        metric=metric_name,
        before_stats=before_stats,
        after_stats=after_stats,
        absolute_change=abs_change,
        percent_change=percent_change,
        median_change=median_change,
        std_ratio=std_ratio,
        cohen_d=cohen_d,
        effect_size_category=effect_category,
        t_statistic=t_stat,
        p_value=p_val,
        mann_whitney_p_value=mw_p,
        is_statistically_significant=is_sig,
        severity=severity,
        summary_sentence=summary,
    )


def analyze_change_impact(
    df: pd.DataFrame,
    change_index: int,
    metric_cols: list[str] | None = None,
    timestamp_col: str | None = None,
    window_size: int | None = None,
) -> ChangeImpactReport:
    """
    Perform pre/post impact analysis at a specific change point across all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing timestamps and metric columns.
    change_index : int
        Observation index where the change begins.
    metric_cols : list[str], optional
        List of numeric metric column names to analyze (auto-detected if None).
    timestamp_col : str, optional
        Timestamp column name (auto-detected if None).
    window_size : int, optional
        If provided, limits the before and after windows to `window_size` points
        immediately surrounding `change_index`. If None, compares all points
        before against all points after.

    Returns
    -------
    ChangeImpactReport
        Comprehensive analysis containing pre/post stats, effect sizes, and severities.
    """
    if change_index <= 0 or change_index >= len(df):
        raise ValueError(
            f"change_index ({change_index}) must be between 1 and {len(df) - 1}."
        )

    # Detect timestamp column
    ts_name = None
    try:
        ts_name = detect_timestamp_column(df, explicit_col=timestamp_col)
    except Exception:
        ts_name = None

    # Detect metric columns
    if metric_cols is None:
        metric_cols = detect_metric_columns(
            df,
            timestamp_col=ts_name if ts_name else "",
            exclude_flags=True,
        )

    if not metric_cols:
        raise ValueError("No numeric metric columns available for impact analysis.")

    # Slice before and after data
    if window_size is not None and window_size > 0:
        start_idx = max(0, change_index - window_size)
        end_idx = min(len(df), change_index + window_size)
        before_df = df.iloc[start_idx:change_index]
        after_df = df.iloc[change_index:end_idx]
    else:
        before_df = df.iloc[:change_index]
        after_df = df.iloc[change_index:]

    metrics_impact: dict[str, MetricImpact] = {}
    max_severity = "negligible"

    for col in metric_cols:
        impact = analyze_metric_shift(
            before_series=before_df[col],
            after_series=after_df[col],
            metric_name=col,
        )
        metrics_impact[col] = impact

        if SEVERITY_RANK[impact.severity] > SEVERITY_RANK[max_severity]:
            max_severity = impact.severity

    # Extract timestamp if available
    change_ts: pd.Timestamp | None = None
    if ts_name and ts_name in df.columns:
        val = df[ts_name].iloc[change_index]
        try:
            change_ts = pd.to_datetime(val)
        except Exception:
            change_ts = None

    return ChangeImpactReport(
        change_index=change_index,
        timestamp=change_ts,
        window_size=window_size,
        metrics=metrics_impact,
        overall_severity=max_severity,
    )


def analyze_all_changes(
    df: pd.DataFrame,
    change_indices: list[int],
    metric_cols: list[str] | None = None,
    timestamp_col: str | None = None,
    window_size: int | None = None,
) -> list[ChangeImpactReport]:
    """
    Analyze impact for a list of detected change indices.
    """
    reports = []
    for idx in sorted(set(change_indices)):
        if 0 < idx < len(df):
            report = analyze_change_impact(
                df=df,
                change_index=idx,
                metric_cols=metric_cols,
                timestamp_col=timestamp_col,
                window_size=window_size,
            )
            reports.append(report)
    return reports
