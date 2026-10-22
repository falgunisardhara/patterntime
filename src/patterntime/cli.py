"""
PatternTime Command-Line Interface (CLI).

Provides commands for database operations, CSV telemetry validation,
change point detection, statistical impact analysis, and historical run inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from patterntime.detectors import (
    CusumDetector,
    DetectionResult,
    EWMADetector,
    RobustCusumDetector,
    SlidingWindowDetector,
    get_detector,
    list_available_detectors,
)
from patterntime.impact import analyze_all_changes, analyze_change_impact
from patterntime.storage import DatabaseStorage, get_db_url
from patterntime.validator import load_and_validate_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for patterntime CLI."""
    parser = argparse.ArgumentParser(
        prog="patterntime",
        description="PatternTime: Interactive AI/ML Time-Series Investigation Laboratory",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # -----------------------------------------------------------------------
    # DB commands
    # -----------------------------------------------------------------------
    db_parser = subparsers.add_parser("db", help="Database management and connection checks")
    db_sub = db_parser.add_subparsers(dest="db_action", help="Database action")

    db_sub.add_parser("test", help="Test MySQL connection configured in .env")
    db_sub.add_parser("init", help="Create/verify database tables in MySQL")
    db_sub.add_parser("status", help="Show connection status and summary statistics")

    # -----------------------------------------------------------------------
    # Validate command
    # -----------------------------------------------------------------------
    val_parser = subparsers.add_parser("validate", help="Validate a time-series CSV file")
    val_parser.add_argument("csv_path", type=str, help="Path to the telemetry CSV file")
    val_parser.add_argument("--timestamp-col", type=str, default=None, help="Name of timestamp column")
    val_parser.add_argument("--json", action="store_true", help="Output validation result as JSON")

    # -----------------------------------------------------------------------
    # Detect command
    # -----------------------------------------------------------------------
    det_parser = subparsers.add_parser("detect", help="Run change point detection on a CSV file")
    det_parser.add_argument("csv_path", type=str, help="Path to the telemetry CSV file")
    det_parser.add_argument("--metric", type=str, default=None, help="Metric column to analyze")
    det_parser.add_argument(
        "--algorithm",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "cusum", "robust_cusum", "ewma"],
        help="Detection algorithm to use (default: sliding_window)",
    )
    det_parser.add_argument("--window-size", type=int, default=30, help="Sliding window size")
    det_parser.add_argument("--threshold", type=float, default=2.0, help="Detection threshold")
    det_parser.add_argument("--save", action="store_true", default=True, help="Save detection run to MySQL (default: True)")
    det_parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save run to database")
    det_parser.add_argument("--json", action="store_true", help="Output detection result as JSON")

    # -----------------------------------------------------------------------
    # Runs commands
    # -----------------------------------------------------------------------
    runs_parser = subparsers.add_parser("runs", help="Inspect historical runs in MySQL")
    runs_sub = runs_parser.add_subparsers(dest="runs_action", help="Runs action")

    list_runs_parser = runs_sub.add_parser("list", help="List recent detection runs")
    list_runs_parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    list_runs_parser.add_argument("--limit", type=int, default=20, help="Max runs to return (default: 20)")
    list_runs_parser.add_argument("--json", action="store_true", help="Output as JSON")

    show_run_parser = runs_sub.add_parser("show", help="Show full details of a specific detection run")
    show_run_parser.add_argument("run_id", type=int, help="ID of the detection run")
    show_run_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # -----------------------------------------------------------------------
    # Datasets commands
    # -----------------------------------------------------------------------
    ds_parser = subparsers.add_parser("datasets", help="Inspect datasets in MySQL")
    ds_sub = ds_parser.add_subparsers(dest="ds_action", help="Datasets action")
    list_ds_parser = ds_sub.add_parser("list", help="List all registered datasets")
    list_ds_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # -----------------------------------------------------------------------
    # Profile command (Per-Endpoint decomposition)
    # -----------------------------------------------------------------------
    prof_parser = subparsers.add_parser("profile", help="Decompose logs by API endpoint and isolate regressions")
    prof_parser.add_argument("csv_path", type=str, help="Path to telemetry CSV file")
    prof_parser.add_argument("--group-by", type=str, default=None, help="Categorical column to group by (default: auto-detected, e.g. api_name)")
    prof_parser.add_argument("--metric", type=str, default=None, help="Metric column to profile (default: auto-detected, e.g. execution_time_ms)")
    prof_parser.add_argument("--status-col", type=str, default=None, help="Status code column for error rate calculations")
    prof_parser.add_argument(
        "--algorithm",
        type=str,
        default="ewma",
        choices=["sliding_window", "cusum", "robust_cusum", "ewma"],
        help="Detection algorithm per endpoint (default: ewma)",
    )
    prof_parser.add_argument("--min-samples", type=int, default=15, help="Minimum requests required to run change detection (default: 15)")
    prof_parser.add_argument("--json", action="store_true", help="Output profile report as JSON")

    # -----------------------------------------------------------------------
    # Gate command (CI/CD Performance Regression Quality Gate)
    # -----------------------------------------------------------------------
    gate_parser = subparsers.add_parser(
        "gate",
        help="CI/CD Performance Quality Gate: evaluate telemetry against baselines/SLAs",
    )
    gate_parser.add_argument("candidate_path", type=str, help="Path to candidate telemetry CSV file")
    gate_parser.add_argument("--baseline", type=str, default=None, help="Path to baseline telemetry CSV file")
    gate_parser.add_argument("--metric", type=str, default=None, help="Target metric column (auto-detected if omitted)")
    gate_parser.add_argument("--max-cohen-d", type=float, default=0.8, help="Maximum allowed Cohen's d effect size (default: 0.8)")
    gate_parser.add_argument("--max-pct-increase", type=float, default=20.0, help="Maximum allowed latency percent increase (default: 20.0)")
    gate_parser.add_argument("--max-error-rate", type=float, default=5.0, help="Maximum allowed error rate percent (default: 5.0)")
    gate_parser.add_argument(
        "--fail-on",
        type=str,
        default="critical,high",
        help="Comma-separated severity levels that fail the gate (default: critical,high)",
    )
    gate_parser.add_argument("--per-endpoint", action="store_true", help="Profile and enforce quality gate across individual API routes")
    gate_parser.add_argument("--group-by", type=str, default=None, help="Route dimension column (auto-detected if omitted)")
    gate_parser.add_argument("--status-col", type=str, default=None, help="Status code column (auto-detected if omitted)")
    gate_parser.add_argument(
        "--algorithm",
        type=str,
        default="ewma",
        choices=["sliding_window", "cusum", "robust_cusum", "ewma"],
        help="Detection algorithm for in-series checks (default: ewma)",
    )
    gate_parser.add_argument("--threshold", type=float, default=None, help="Detection threshold/control limit override")
    gate_parser.add_argument("--min-samples", type=int, default=15, help="Minimum requests required to profile route (default: 15)")
    gate_parser.add_argument(
        "--output-summary",
        type=str,
        nargs="?",
        const="patterntime_summary.md",
        default=None,
        help="Path to output GitHub Step Summary markdown file (defaults to patterntime_summary.md or $GITHUB_STEP_SUMMARY)",
    )
    gate_parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json", "markdown"],
        help="Output format on stdout (default: text)",
    )
    gate_parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output")
    gate_parser.add_argument("--quiet", action="store_true", help="Suppress non-error stdout (only exit code)")

    # -----------------------------------------------------------------------
    # Serve command (local API and UI server)
    # -----------------------------------------------------------------------
    serve_parser = subparsers.add_parser("serve", help="Start local API and static web server")
    serve_parser.add_argument("--port", type=int, default=8089, help="Port to listen on (default: 8089)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")

    return parser


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------
def handle_db(args: argparse.Namespace) -> int:
    """Handle database actions (test, init, status)."""
    action = args.db_action or "status"
    storage = DatabaseStorage(auto_create_tables=False)

    if action == "test":
        ok, msg = storage.test_connection()
        if ok:
            print(f"[OK] {msg}")
            return 0
        print(f"[ERROR] {msg}", file=sys.stderr)
        return 1

    if action == "init":
        try:
            storage.init_db()
            print("[OK] Database tables verified / created successfully.")
            return 0
        except Exception as err:
            print(f"[ERROR] Table initialization failed: {err}", file=sys.stderr)
            return 1

    if action == "status":
        ok, msg = storage.test_connection()
        print("PatternTime Database Status:")
        print(f"  Connection URL: {storage.db_url.split('@')[-1] if '@' in storage.db_url else storage.db_url}")
        print(f"  Connected:      {'YES' if ok else 'NO'}")
        if not ok:
            print(f"  Error:          {msg}")
            return 1

        try:
            from sqlalchemy import inspect
            inspector = inspect(storage.engine)
            tables = inspector.get_table_names()
            datasets = storage.list_datasets()
            runs = storage.list_runs(limit=100)
            print(f"  Tables:         {', '.join(tables)}")
            print(f"  Datasets:       {len(datasets)}")
            print(f"  Detection Runs: {len(runs)}")
            return 0
        except Exception as err:
            print(f"  Error reading status: {err}", file=sys.stderr)
            return 1

    return 0


def handle_validate(args: argparse.Namespace) -> int:
    """Handle telemetry CSV validation."""
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"[ERROR] File not found: {csv_file}", file=sys.stderr)
        return 1

    try:
        from patterntime.validator import TimeSeriesValidationError, ValidationConfig
        config = ValidationConfig(timestamp_col=args.timestamp_col)
        result = load_and_validate_csv(csv_file, config=config)
    except TimeSeriesValidationError as err:
        if args.json:
            print(json.dumps({"valid": False, "error": str(err)}, indent=2))
        else:
            print(f"[ERROR] Validation failed: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Failed to read CSV: {err}", file=sys.stderr)
        return 1

    if args.json:
        out = result.to_dict()
        out["valid"] = True
        print(json.dumps(out, indent=2))
        return 0

    print(f"Validation Result for: {csv_file.name}")
    print(f"  Status:            VALID")
    print(f"  Rows:              {result.total_rows}")
    print(f"  Timestamp Column:  {result.timestamp_col}")
    print(f"  Metric Columns:    {', '.join(result.metric_cols)}")
    print(f"  Inferred Cadence:  {result.inferred_frequency or 'irregular'}")
    print(f"  Time Range:        {result.time_range[0]} to {result.time_range[1]}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    return 0


def handle_detect(args: argparse.Namespace) -> int:
    """Handle change detection and persistence."""
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"[ERROR] File not found: {csv_file}", file=sys.stderr)
        return 1

    try:
        from patterntime.validator import TimeSeriesValidationError
        val = load_and_validate_csv(csv_file)
    except TimeSeriesValidationError as err:
        print(f"[ERROR] Validation failed: {err}", file=sys.stderr)
        return 1

    df = val.df
    metric = args.metric or (val.metric_cols[0] if val.metric_cols else None)
    if not metric or metric not in df.columns:
        print(f"[ERROR] Metric column '{metric}' not found in CSV.", file=sys.stderr)
        return 1

    # Configure algorithm
    params = {}
    if args.algorithm == "sliding_window":
        detector = SlidingWindowDetector(window_size=args.window_size, threshold=args.threshold)
        params = {"window_size": args.window_size, "threshold": args.threshold}
    elif args.algorithm == "cusum":
        detector = CusumDetector(threshold=args.threshold)
        params = {"threshold": args.threshold}
    elif args.algorithm == "robust_cusum":
        detector = RobustCusumDetector(threshold=args.threshold)
        params = {"threshold": args.threshold}
    elif args.algorithm == "ewma":
        detector = EWMADetector(control_limit=args.threshold)
        params = {"control_limit": args.threshold}
    else:
        print(f"[ERROR] Unsupported algorithm: {args.algorithm}", file=sys.stderr)
        return 1

    series = df[metric]
    timestamps = df[val.timestamp_col] if val.timestamp_col and val.timestamp_col in df.columns else None
    det_result = detector.detect(series=series, timestamps=timestamps)

    # Analyze impact of changes
    impact_reports = []
    if det_result.events:
        impact_reports = analyze_all_changes(
            df=df,
            change_indices=det_result.indices,
            metric_cols=val.metric_cols,
            timestamp_col=val.timestamp_col,
            window_size=args.window_size,
        )

    run_id = None
    if args.save:
        try:
            storage = DatabaseStorage()
            ds_id = storage.save_dataset(
                name=csv_file.name,
                row_count=len(df),
                cadence=val.inferred_frequency,
                timestamp_col=val.timestamp_col,
                metric_cols=val.metric_cols,
            )
            first_report = impact_reports[0] if impact_reports else None
            run_id = storage.save_run(
                dataset_name=csv_file.name,
                metric_name=metric,
                detection_result=det_result,
                impact_report=first_report,
                dataset_id=ds_id,
            )
        except Exception as err:
            print(f"[WARNING] Database save failed: {err}", file=sys.stderr)

    if args.json:
        out = {
            "dataset": csv_file.name,
            "metric": metric,
            "algorithm": args.algorithm,
            "params": params,
            "run_id": run_id,
            "total_events": len(det_result.events),
            "execution_time_ms": det_result.execution_time_ms,
            "events": [e.to_dict() for e in det_result.events],
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"Detection Results: {csv_file.name}")
    print(f"  Metric:             {metric}")
    print(f"  Algorithm:          {args.algorithm}")
    print(f"  Total Events:       {len(det_result.events)}")
    print(f"  Execution Time:     {det_result.execution_time_ms:.2f} ms")
    if run_id:
        print(f"  Saved to MySQL Run: #{run_id}")

    if det_result.events:
        print("\nDetected Change Points:")
        for idx, ev in enumerate(det_result.events[:10], 1):
            ts = f" at {ev.timestamp}" if ev.timestamp is not None else ""
            print(f"  {idx}. Index {ev.index}{ts} | Direction: {ev.direction.upper()} | Score: {ev.score:.2f}")
        if len(det_result.events) > 10:
            print(f"  ... and {len(det_result.events) - 10} more events")

    return 0


def handle_runs(args: argparse.Namespace) -> int:
    """Handle listing or showing saved detection runs."""
    action = args.runs_action or "list"
    storage = DatabaseStorage(auto_create_tables=False)

    if action == "list":
        runs = storage.list_runs(dataset_name=args.dataset, limit=args.limit)
        if args.json:
            print(json.dumps(runs, indent=2))
            return 0

        if not runs:
            print("No detection runs found.")
            return 0

        print(f"{'ID':<6} {'DATASET':<26} {'METRIC':<18} {'ALGO':<16} {'EVENTS':<8} {'CREATED'}")
        print("-" * 90)
        for r in runs:
            created = r.get("created_at", "")[:19] if r.get("created_at") else ""
            print(f"{r['id']:<6} {r['dataset_name']:<26} {r['metric_name']:<18} {r['algorithm_name']:<16} {r['total_events']:<8} {created}")
        return 0

    if action == "show":
        run = storage.get_run(args.run_id)
        if not run:
            print(f"[ERROR] Run #{args.run_id} not found.", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(run, indent=2))
            return 0

        print(f"=== Detection Run #{run['id']} ===")
        print(f"  Dataset:         {run['dataset_name']}")
        print(f"  Metric:          {run['metric_name']}")
        print(f"  Algorithm:       {run['algorithm_name']}")
        print(f"  Parameters:      {run['params']}")
        print(f"  Execution Time:  {run['execution_time_ms']} ms")
        print(f"  Total Events:    {run['total_events']}")
        print(f"  Created:         {run['created_at']}")

        events = run.get("events", [])
        if events:
            print(f"\nChange Points ({len(events)}):")
            for e in events[:10]:
                ts = f" ({e['event_timestamp']})" if e.get("event_timestamp") else ""
                print(f"  - Obs {e['observation_index']}{ts}: {e['direction'].upper()} (Score: {e['score']})")
            if len(events) > 10:
                print(f"  ... and {len(events) - 10} more events")

        reports = run.get("impact_reports", [])
        if reports:
            print(f"\nStatistical Impact Reports ({len(reports)}):")
            for rep in reports:
                pct = f"{rep['percent_change']:+.1f}%" if rep.get("percent_change") is not None else "N/A"
                print(f"  - Change @ {rep['change_index']} | Severity: {rep['severity'].upper()} | Cohen's d: {rep['cohen_d']:.2f} ({rep['effect_size_category']})")
                print(f"    Pre-mean: {rep['pre_mean']:.2f} -> Post-mean: {rep['post_mean']:.2f} (Shift: {pct})")
                if rep.get("summary_sentence"):
                    print(f"    Summary: {rep['summary_sentence']}")

        return 0

    return 0


def handle_datasets(args: argparse.Namespace) -> int:
    """Handle listing registered datasets."""
    action = args.ds_action or "list"
    storage = DatabaseStorage(auto_create_tables=False)

    if action == "list":
        datasets = storage.list_datasets()
        if args.json:
            print(json.dumps(datasets, indent=2))
            return 0

        if not datasets:
            print("No registered datasets found in database.")
            return 0

        print(f"{'ID':<6} {'NAME':<28} {'ROWS':<8} {'CADENCE':<10} {'TIMESTAMP COL':<16} {'METRICS'}")
        print("-" * 90)
        for d in datasets:
            metrics_str = ", ".join(d.get("metric_cols", []))
            print(f"{d['id']:<6} {d['name']:<28} {d['row_count']:<8} {str(d.get('cadence') or '-'):<10} {str(d.get('timestamp_col') or '-'):<16} {metrics_str}")
        return 0

    return 0


def handle_profile(args: argparse.Namespace) -> int:
    """Handle per-endpoint API telemetry profiling."""
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"[ERROR] File not found: {csv_file}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(csv_file)
    except Exception as err:
        print(f"[ERROR] Could not read CSV: {err}", file=sys.stderr)
        return 1

    try:
        from patterntime.profiler import profile_endpoints
        report = profile_endpoints(
            df=df,
            group_col=args.group_by,
            metric_col=args.metric,
            status_col=args.status_col,
            algorithm=args.algorithm,
            min_samples=args.min_samples,
            dataset_name=csv_file.name,
        )
    except Exception as err:
        print(f"[ERROR] Profiling failed: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print(f"=== API Telemetry Profile: {csv_file.name} ===")
    print(f"  Grouped By:          {report.group_col}")
    print(f"  Metric Profiled:     {report.metric_col}")
    print(f"  Total Requests:      {report.total_requests:,}")
    print(f"  Unique Endpoints:    {report.total_endpoints}")
    print(f"  Regressed Routes:    {len(report.regressed_endpoints)}")

    print("\n" + f"{'ENDPOINT':<38} {'CALLS':<8} {'ERR%':<6} {'MEAN':<8} {'P95':<8} {'SHIFTS':<8} {'MAX D':<8} {'SEVERITY'}")
    print("-" * 102)

    for ep in report.endpoints:
        sev = ep.max_severity.upper() if ep.max_severity != "none" else "STABLE"
        err_str = f"{ep.error_rate_pct:.1f}%" if ep.error_count > 0 else "0.0%"
        d_str = f"{ep.max_cohen_d:.2f}" if ep.max_cohen_d > 0 else "-"
        name_trunc = (ep.endpoint_name[:35] + "...") if len(ep.endpoint_name) > 38 else ep.endpoint_name
        print(f"{name_trunc:<38} {ep.request_count:<8} {err_str:<6} {ep.mean_latency:<8.1f} {ep.p95_latency:<8.1f} {ep.total_events:<8} {d_str:<8} {sev}")

    if report.regressed_endpoints:
        print("\n🚨 Identified Performance Regressions:")
        for r_ep in report.regressed_endpoints[:5]:
            print(f"  - [{r_ep.max_severity.upper()}] {r_ep.endpoint_name}: {r_ep.summary}")

    return 0


def handle_gate(args: argparse.Namespace) -> int:
    """Handle CI/CD performance regression quality gate evaluation."""
    from patterntime.gate import GateRuleConfig, evaluate_gate

    candidate_file = Path(args.candidate_path)
    if not candidate_file.exists():
        print(f"[ERROR] Candidate file '{args.candidate_path}' not found.", file=sys.stderr)
        return 2

    try:
        df_cand = pd.read_csv(candidate_file)
    except Exception as err:
        print(f"[ERROR] Failed to read candidate CSV: {err}", file=sys.stderr)
        return 2

    df_base = None
    baseline_name = None
    if args.baseline:
        base_file = Path(args.baseline)
        if not base_file.exists():
            print(f"[ERROR] Baseline file '{args.baseline}' not found.", file=sys.stderr)
            return 2
        try:
            df_base = pd.read_csv(base_file)
            baseline_name = base_file.name
        except Exception as err:
            print(f"[ERROR] Failed to read baseline CSV: {err}", file=sys.stderr)
            return 2

    severities = [s.strip().lower() for s in args.fail_on.split(",") if s.strip()]

    config = GateRuleConfig(
        metric=args.metric,
        baseline_path=args.baseline,
        max_cohen_d=args.max_cohen_d,
        max_pct_increase=args.max_pct_increase,
        max_error_rate_pct=args.max_error_rate,
        fail_on_severity=severities,
        per_endpoint=args.per_endpoint,
        group_col=args.group_by,
        status_col=args.status_col,
        algorithm=args.algorithm,
        threshold=args.threshold,
        min_samples=args.min_samples,
    )

    try:
        report = evaluate_gate(
            df=df_cand,
            dataset_name=candidate_file.name,
            config=config,
            df_baseline=df_base,
            baseline_name=baseline_name,
        )
    except Exception as err:
        print(f"[ERROR] Gate evaluation failed: {err}", file=sys.stderr)
        return 2

    # Write GitHub Actions Step Summary if requested or if GITHUB_STEP_SUMMARY env var is set
    summary_target = args.output_summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_target:
        summary_path = Path(summary_target)
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(report.to_markdown(), encoding="utf-8")
        except Exception as err:
            print(f"[WARNING] Failed to write summary markdown to '{summary_target}': {err}", file=sys.stderr)

    if not args.quiet:
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        elif args.format == "markdown":
            print(report.to_markdown())
        else:
            use_color = not args.no_color and sys.stdout.isatty()
            print(report.to_terminal_text(use_color=use_color))

    # Exit code: 0 if passed, 1 if failed
    return 0 if report.passed else 1


def handle_serve(args: argparse.Namespace) -> int:
    """Start local API and UI server."""
    from patterntime.server import run_server
    try:
        run_server(host=args.host, port=args.port)
        return 0
    except Exception as err:
        print(f"[ERROR] Failed to start server: {err}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the patterntime CLI application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "db":
        return handle_db(args)
    if args.subcommand == "validate":
        return handle_validate(args)
    if args.subcommand == "detect":
        return handle_detect(args)
    if args.subcommand == "runs":
        return handle_runs(args)
    if args.subcommand == "datasets":
        return handle_datasets(args)
    if args.subcommand == "profile":
        return handle_profile(args)
    if args.subcommand == "gate":
        return handle_gate(args)
    if args.subcommand == "serve":
        return handle_serve(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
