"""
Per-Endpoint API Telemetry Profiler for PatternTime.

Decomposes heterogeneous API logs (e.g., CakePHP, FastAPI, Express, Django)
into individual endpoint cohorts, computes reliability metrics (p50, p95, p99, error rates),
and automatically runs structural change detection across each route to isolate performance regressions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from patterntime.detectors import BaseDetector, get_detector
from patterntime.impact import SEVERITY_RANK, analyze_change_impact
from patterntime.validator import detect_metric_columns, detect_timestamp_column

# Common categorical column candidates used for API endpoint names or routes
COMMON_GROUP_NAMES = [
    "api_name",
    "endpoint",
    "route",
    "url",
    "path",
    "api",
    "controller",
    "action",
    "method",
    "service",
    "request_uri",
    "handler",
    "name",
]

COMMON_STATUS_NAMES = [
    "status_code",
    "status",
    "code",
    "http_status",
    "response_code",
]


def find_grouping_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify potential categorical grouping columns (e.g. API name, endpoint) in a DataFrame.
    """
    candidates = []
    lower_map = {col.lower(): col for col in df.columns}

    # 1. Check known naming conventions
    for name in COMMON_GROUP_NAMES:
        if name in lower_map:
            col = lower_map[name]
            if col not in candidates:
                candidates.append(col)

    # 2. Check object/string columns with reasonable cardinality (2 to 500 distinct values)
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if col not in candidates:
            distinct_count = df[col].nunique(dropna=True)
            if 1 < distinct_count <= 1000:
                candidates.append(col)

    return candidates


def find_status_column(df: pd.DataFrame) -> str | None:
    """Find HTTP status code column if present in DataFrame."""
    lower_map = {col.lower(): col for col in df.columns}
    for name in COMMON_STATUS_NAMES:
        if name in lower_map:
            return lower_map[name]
    return None


@dataclass
class EndpointProfile:
    """Statistical profile and change analysis for a single API endpoint."""

    endpoint_name: str
    request_count: int
    error_count: int
    error_rate_pct: float
    mean_latency: float
    median_latency: float
    std_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    total_events: int
    max_severity: str  # "critical", "high", "medium", "low", "negligible", "none"
    max_cohen_d: float
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert EndpointProfile to dictionary."""
        return {
            "endpoint_name": self.endpoint_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate_pct": round(self.error_rate_pct, 2),
            "mean_latency": round(self.mean_latency, 2),
            "median_latency": round(self.median_latency, 2),
            "std_latency": round(self.std_latency, 2),
            "p95_latency": round(self.p95_latency, 2),
            "p99_latency": round(self.p99_latency, 2),
            "min_latency": round(self.min_latency, 2),
            "max_latency": round(self.max_latency, 2),
            "total_events": self.total_events,
            "max_severity": self.max_severity,
            "max_cohen_d": round(self.max_cohen_d, 2),
            "events": self.events,
            "summary": self.summary,
        }


@dataclass
class ProfileReport:
    """Comprehensive multi-endpoint profile and regression report."""

    dataset_name: str
    group_col: str
    metric_col: str
    total_requests: int
    total_endpoints: int
    endpoints: list[EndpointProfile]
    regressed_endpoints: list[EndpointProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "group_col": self.group_col,
            "metric_col": self.metric_col,
            "total_requests": self.total_requests,
            "total_endpoints": self.total_endpoints,
            "regressed_count": len(self.regressed_endpoints),
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "regressed_endpoints": [ep.to_dict() for ep in self.regressed_endpoints],
        }


def profile_endpoints(
    df: pd.DataFrame,
    group_col: str | None = None,
    metric_col: str | None = None,
    status_col: str | None = None,
    timestamp_col: str | None = None,
    algorithm: str = "ewma",
    min_samples: int = 15,
    dataset_name: str = "dataset.csv",
) -> ProfileReport:
    """
    Partition telemetry data by API endpoint, compute performance distributions,
    and detect structural regime changes per endpoint.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Auto-resolve grouping column
    if not group_col:
        candidates = find_grouping_columns(df)
        if not candidates:
            raise ValueError("No categorical column (e.g. api_name, endpoint, route) found for grouping.")
        group_col = candidates[0]
    elif group_col not in df.columns:
        raise ValueError(f"Specified grouping column '{group_col}' not found in DataFrame.")

    # Auto-resolve timestamp column first
    if not timestamp_col:
        try:
            timestamp_col = detect_timestamp_column(df)
        except Exception:
            timestamp_col = None

    # Auto-resolve metric column
    if not metric_col:
        metrics = detect_metric_columns(df, timestamp_col=timestamp_col or "")
        if not metrics:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                metrics = num_cols
            else:
                raise ValueError("No numeric metric columns found in DataFrame.")
        metric_col = metrics[0]
    elif metric_col not in df.columns:
        raise ValueError(f"Specified metric column '{metric_col}' not found in DataFrame.")

    # Auto-resolve status code column
    if status_col is None:
        status_col = find_status_column(df)

    # Sort DataFrame chronologically if timestamp exists
    working_df = df.copy()
    if timestamp_col and timestamp_col in working_df.columns:
        try:
            working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col], errors="coerce")
            working_df = working_df.sort_values(by=timestamp_col).reset_index(drop=True)
        except Exception:
            pass

    # Instantiate detector
    try:
        detector = get_detector(algorithm)
    except Exception:
        detector = get_detector("ewma")

    profiles: list[EndpointProfile] = []

    # Group by endpoint
    grouped = working_df.groupby(group_col, dropna=False)

    for name, group in grouped:
        endpoint_name = str(name) if pd.notna(name) else "unknown"
        sub_df = group.dropna(subset=[metric_col]).reset_index(drop=True)
        n_requests = len(sub_df)

        if n_requests < 2:
            continue

        series = sub_df[metric_col].astype(float)
        m_mean = float(series.mean())
        m_median = float(series.median())
        m_std = float(series.std()) if n_requests > 1 else 0.0
        m_min = float(series.min())
        m_max = float(series.max())
        m_p95 = float(series.quantile(0.95))
        m_p99 = float(series.quantile(0.99))

        # Calculate error rate if status code column exists
        n_errors = 0
        error_rate = 0.0
        if status_col and status_col in sub_df.columns:
            status_series = pd.to_numeric(sub_df[status_col], errors="coerce")
            n_errors = int((status_series >= 400).sum())
            error_rate = (n_errors / n_requests) * 100.0

        # Change detection if sample size meets threshold
        events_list = []
        max_severity = "none"
        max_cohen_d = 0.0
        summary_sentence = ""

        if n_requests >= min_samples:
            try:
                ts_series = sub_df[timestamp_col] if timestamp_col and timestamp_col in sub_df.columns else None
                det_result = detector.detect(series, timestamps=ts_series)

                if det_result.events:
                    for ev in det_result.events:
                        ev_dict = ev.to_dict()
                        # Add impact metrics
                        try:
                            impact = analyze_change_impact(
                                df=sub_df,
                                change_index=ev.index,
                                metric_cols=[metric_col],
                                timestamp_col=timestamp_col,
                            )
                            metric_imp = impact.metrics.get(metric_col)
                            if metric_imp:
                                ev_dict["severity"] = metric_imp.severity
                                ev_dict["cohen_d"] = round(metric_imp.cohen_d, 2)
                                ev_dict["pre_mean"] = round(metric_imp.before_stats.mean, 2)
                                ev_dict["post_mean"] = round(metric_imp.after_stats.mean, 2)
                                ev_dict["percent_change"] = (
                                    round(metric_imp.percent_change, 1)
                                    if metric_imp.percent_change is not None
                                    else None
                                )
                                ev_dict["summary"] = metric_imp.summary_sentence

                                # Track highest severity
                                if SEVERITY_RANK[metric_imp.severity] > SEVERITY_RANK.get(max_severity, 0):
                                    max_severity = metric_imp.severity
                                    max_cohen_d = abs(metric_imp.cohen_d)
                                    summary_sentence = metric_imp.summary_sentence
                        except Exception:
                            pass

                        events_list.append(ev_dict)
            except Exception:
                pass

        if not summary_sentence:
            if max_severity != "none":
                summary_sentence = f"Regime shift detected ({max_severity.upper()}) with {len(events_list)} change boundary."
            else:
                summary_sentence = f"Stable performance across {n_requests} requests (median: {m_median:.1f}ms)."

        profiles.append(
            EndpointProfile(
                endpoint_name=endpoint_name,
                request_count=n_requests,
                error_count=n_errors,
                error_rate_pct=error_rate,
                mean_latency=m_mean,
                median_latency=m_median,
                std_latency=m_std,
                p95_latency=m_p95,
                p99_latency=m_p99,
                min_latency=m_min,
                max_latency=m_max,
                total_events=len(events_list),
                max_severity=max_severity,
                max_cohen_d=max_cohen_d,
                events=events_list,
                summary=summary_sentence,
            )
        )

    # Sort all profiles by request count descending
    profiles.sort(key=lambda p: p.request_count, reverse=True)

    # Filter regressed endpoints and sort by severity rank and Cohen's d descending
    regressed = [
        p for p in profiles if p.total_events > 0 and SEVERITY_RANK.get(p.max_severity, 0) > 0
    ]
    regressed.sort(
        key=lambda p: (SEVERITY_RANK.get(p.max_severity, 0), p.max_cohen_d),
        reverse=True,
    )

    return ProfileReport(
        dataset_name=dataset_name,
        group_col=group_col,
        metric_col=metric_col,
        total_requests=len(working_df),
        total_endpoints=len(profiles),
        endpoints=profiles,
        regressed_endpoints=regressed,
    )
