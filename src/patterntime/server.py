"""
Local API and Static Web Server for PatternTime.

Provides REST API endpoints for MySQL persistence and serves the interactive
visualization workbench with zero external dependencies (built on standard library http.server).
"""

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from patterntime.detectors import (
    ChangeEvent,
    DetectionResult,
    get_detector,
)
from patterntime.storage import DatabaseStorage


class PatternTimeHandler(SimpleHTTPRequestHandler):
    """HTTP request handler supporting both static files and REST API routes."""

    def __init__(self, *args, directory: str | None = None, **kwargs):
        if directory is None:
            # Default directory is workspace root
            directory = str(Path(__file__).resolve().parent.parent.parent)
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        """Add permissive CORS and caching headers for local development."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for static assets and API routes."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Redirect root to visualization workbench
        if path == "/" or path == "":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/visualization/index.html")
            self.end_headers()
            return

        # Route API calls
        if path.startswith("/api/"):
            self._handle_api_get(path, parse_qs(parsed.query))
            return

        # Fallback to standard static file server
        super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests for API routes."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
            except Exception as err:
                self._send_json({"error": f"Invalid JSON payload: {err}"}, status=HTTPStatus.BAD_REQUEST)
                return

            self._handle_api_post(path, data)
            return

        self._send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        """Handle DELETE requests for API routes."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/runs/"):
            run_id_str = path.split("/api/runs/")[-1]
            try:
                run_id = int(run_id_str)
                storage = DatabaseStorage(auto_create_tables=False)
                deleted = storage.delete_run(run_id)
                if deleted:
                    self._send_json({"status": "success", "message": f"Run #{run_id} deleted."})
                else:
                    self._send_json({"error": f"Run #{run_id} not found."}, status=HTTPStatus.NOT_FOUND)
            except ValueError:
                self._send_json({"error": "Invalid run ID format"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as err:
                self._send_json({"error": str(err)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    # -----------------------------------------------------------------------
    # API Route Implementations
    # -----------------------------------------------------------------------
    def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        storage = DatabaseStorage(auto_create_tables=False)

        if path == "/api/db/status":
            ok, msg = storage.test_connection()
            tables = []
            if ok:
                try:
                    from sqlalchemy import inspect
                    tables = inspect(storage.engine).get_table_names()
                except Exception:
                    pass
            self._send_json({
                "connected": ok,
                "message": msg,
                "driver": "mysql+pymysql",
                "tables": tables,
            })
            return

        if path == "/api/datasets":
            try:
                datasets = storage.list_datasets()
                self._send_json({"datasets": datasets})
            except Exception as err:
                self._send_json({"error": str(err)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/runs":
            try:
                ds_filter = query.get("dataset", [None])[0]
                limit = int(query.get("limit", [50])[0])
                runs = storage.list_runs(dataset_name=ds_filter, limit=limit)
                self._send_json({"runs": runs})
            except Exception as err:
                self._send_json({"error": str(err)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path.startswith("/api/runs/"):
            run_id_str = path.split("/api/runs/")[-1]
            try:
                run_id = int(run_id_str)
                run = storage.get_run(run_id)
                if not run:
                    self._send_json({"error": f"Run #{run_id} not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json({"run": run})
            except ValueError:
                self._send_json({"error": "Invalid run ID format"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as err:
                self._send_json({"error": str(err)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": "API route not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str, data: dict[str, Any]) -> None:
        storage = DatabaseStorage()

        if path == "/api/runs":
            try:
                dataset_name = data.get("dataset_name", "unnamed_dataset.csv")
                metric_name = data.get("metric_name", "metric")
                events_data = data.get("events", [])
                algorithm_name = data.get("algorithm_name", "custom")
                params = data.get("params", {})
                exec_time = float(data.get("execution_time_ms", 0.0))
                impact = data.get("impact_report")
                row_count = int(data.get("row_count", 0))

                events = [
                    ChangeEvent(
                        index=e.get("index", 0),
                        timestamp=e.get("timestamp"),
                        direction=e.get("direction", "unknown"),
                        score=float(e.get("score", 0.0)),
                        metadata=e.get("metadata", {}),
                    )
                    for e in events_data
                ]

                det_result = DetectionResult(
                    events=events,
                    algorithm_name=algorithm_name,
                    params=params,
                    execution_time_ms=exec_time,
                )

                ds_id = storage.save_dataset(
                    name=dataset_name,
                    row_count=row_count,
                    timestamp_col=data.get("timestamp_col"),
                    metric_cols=data.get("metric_cols", [metric_name]),
                )

                run_id = storage.save_run(
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    detection_result=det_result,
                    impact_report=impact,
                    dataset_id=ds_id,
                )

                self._send_json({
                    "status": "success",
                    "run_id": run_id,
                    "dataset_id": ds_id,
                    "message": f"Run #{run_id} persisted to MySQL successfully.",
                }, status=HTTPStatus.CREATED)
            except Exception as err:
                self._send_json({"error": str(err)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": "API route not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Helper to serialize response to JSON and send with appropriate headers."""
        response_bytes = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


def run_server(host: str = "0.0.0.0", port: int = 8089) -> None:
    """Start the local HTTP server."""
    server = HTTPServer((host, port), PatternTimeHandler)
    print(f"PatternTime Server running at http://{host}:{port}/")
    print(f"  Interactive Workbench: http://localhost:{port}/visualization/index.html")
    print(f"  Database API Status:   http://localhost:{port}/api/db/status")
    print("Press Ctrl+C to terminate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PatternTime Server...")
        server.server_close()
