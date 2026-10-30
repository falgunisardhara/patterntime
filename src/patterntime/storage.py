"""
Local Storage & Persistence Engine for PatternTime.

Provides relational persistence using MySQL (via SQLAlchemy and PyMySQL),
with environment-variable configuration (.env), automated schema creation,
and full round-trip serialization for datasets, detection runs, change events,
and statistical impact reports.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


# ---------------------------------------------------------------------------
# Database Configuration Helpers
# ---------------------------------------------------------------------------
def get_db_url(env_file: str | Path | None = None) -> str:
    """
    Resolve the database connection URL from environment variables or .env file.
    
    Priority:
    1. Explicit DATABASE_URL environment variable
    2. Composed mysql+pymysql:// URL from DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    3. Fallback to SQLite local file if no MySQL credentials specified
    """
    env_path = Path(env_file) if env_file else Path.cwd() / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)

    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return db_url

    # Check for MySQL parameters
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "3306")
    db_name = os.environ.get("DB_NAME", "patterntime")

    # URL-encode password to handle special symbols cleanly
    encoded_pass = urllib.parse.quote_plus(password) if password else ""
    auth_part = f"{user}:{encoded_pass}" if encoded_pass else user

    return f"mysql+pymysql://{auth_part}@{host}:{port}/{db_name}?charset=utf8mb4"


# ---------------------------------------------------------------------------
# SQLAlchemy Declarative Models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Base declarative class for PatternTime storage."""
    pass


class DatasetModel(Base):
    """Metadata registry of ingested telemetry datasets."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    row_count = Column(Integer, nullable=False, default=0)
    cadence = Column(String(64), nullable=True)
    timestamp_col = Column(String(128), nullable=True)
    metric_cols = Column(Text, nullable=True)  # JSON-encoded column list
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    runs = relationship("DetectionRunModel", back_populates="dataset", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "row_count": self.row_count,
            "cadence": self.cadence,
            "timestamp_col": self.timestamp_col,
            "metric_cols": json.loads(self.metric_cols) if self.metric_cols else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DetectionRunModel(Base):
    """Historical record of an algorithm detection execution."""

    __tablename__ = "detection_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True, index=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    metric_name = Column(String(128), nullable=False, index=True)
    algorithm_name = Column(String(64), nullable=False, index=True)
    params = Column(Text, nullable=True)  # JSON-encoded hyperparameters
    total_events = Column(Integer, nullable=False, default=0)
    execution_time_ms = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    dataset = relationship("DatasetModel", back_populates="runs")
    events = relationship("ChangeEventModel", back_populates="run", cascade="all, delete-orphan")
    impact_reports = relationship("ImpactReportModel", back_populates="run", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "metric_name": self.metric_name,
            "algorithm_name": self.algorithm_name,
            "params": json.loads(self.params) if self.params else {},
            "total_events": self.total_events,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChangeEventModel(Base):
    """Individual structural change point detected in a time series."""

    __tablename__ = "change_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("detection_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    observation_index = Column(Integer, nullable=False)
    event_timestamp = Column(String(64), nullable=True)
    direction = Column(String(16), nullable=False)
    score = Column(Float, nullable=False)
    details = Column(Text, nullable=True)  # JSON-encoded additional info

    run = relationship("DetectionRunModel", back_populates="events")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "observation_index": self.observation_index,
            "event_timestamp": self.event_timestamp,
            "direction": self.direction,
            "score": round(self.score, 3),
            "details": json.loads(self.details) if self.details else {},
        }


class ImpactReportModel(Base):
    """Statistical impact report measuring pre vs. post regime transition."""

    __tablename__ = "impact_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("detection_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    change_index = Column(Integer, nullable=False)
    severity = Column(String(32), nullable=False)
    cohen_d = Column(Float, nullable=False)
    effect_size_category = Column(String(32), nullable=False)
    p_value_welch = Column(Float, nullable=True)
    p_value_mann_whitney = Column(Float, nullable=True)
    pre_mean = Column(Float, nullable=False)
    post_mean = Column(Float, nullable=False)
    absolute_change = Column(Float, nullable=False)
    percent_change = Column(Float, nullable=True)
    pre_stats = Column(Text, nullable=True)  # JSON-encoded SegmentStats
    post_stats = Column(Text, nullable=True)  # JSON-encoded SegmentStats
    summary_sentence = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    run = relationship("DetectionRunModel", back_populates="impact_reports")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "change_index": self.change_index,
            "severity": self.severity,
            "cohen_d": round(self.cohen_d, 3),
            "effect_size_category": self.effect_size_category,
            "p_value_welch": self.p_value_welch,
            "p_value_mann_whitney": self.p_value_mann_whitney,
            "pre_mean": round(self.pre_mean, 3),
            "post_mean": round(self.post_mean, 3),
            "absolute_change": round(self.absolute_change, 3),
            "percent_change": round(self.percent_change, 2) if self.percent_change is not None else None,
            "pre_stats": json.loads(self.pre_stats) if self.pre_stats else {},
            "post_stats": json.loads(self.post_stats) if self.post_stats else {},
            "summary_sentence": self.summary_sentence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Database Storage Manager
# ---------------------------------------------------------------------------
class DatabaseStorage:
    """High-level repository manager for PatternTime persistence."""

    def __init__(self, db_url: str | None = None, auto_create_tables: bool = True):
        self.db_url = db_url or get_db_url()
        self.engine: Engine = create_engine(self.db_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(bind=self.engine)

        if auto_create_tables:
            self.init_db()

    def init_db(self) -> None:
        """Create all required tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    def test_connection(self) -> tuple[bool, str]:
        """Test database connectivity and return status message."""
        try:
            with self.engine.connect() as conn:
                conn.execute(select(1))
            return True, "Database connection successful."
        except Exception as err:
            return False, f"Connection failed: {err}"

    # -----------------------------------------------------------------------
    # Dataset Registry Operations
    # -----------------------------------------------------------------------
    def save_dataset(
        self,
        name: str,
        row_count: int,
        cadence: str | None = None,
        timestamp_col: str | None = None,
        metric_cols: list[str] | None = None,
    ) -> int:
        """Register or update dataset metadata in the database."""
        with self.session_factory() as session:
            dataset = session.query(DatasetModel).filter_by(name=name).first()
            if not dataset:
                dataset = DatasetModel(name=name)
                session.add(dataset)

            dataset.row_count = row_count
            dataset.cadence = cadence
            dataset.timestamp_col = timestamp_col
            dataset.metric_cols = json.dumps(metric_cols or [])
            dataset.updated_at = datetime.now(timezone.utc)

            session.commit()
            return dataset.id

    def get_dataset(self, name_or_id: str | int) -> dict[str, Any] | None:
        """Retrieve dataset details by id or name."""
        with self.session_factory() as session:
            if isinstance(name_or_id, int):
                dataset = session.get(DatasetModel, name_or_id)
            else:
                dataset = session.query(DatasetModel).filter_by(name=str(name_or_id)).first()

            return dataset.to_dict() if dataset else None

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all registered datasets."""
        with self.session_factory() as session:
            records = session.query(DatasetModel).order_by(desc(DatasetModel.updated_at)).all()
            return [r.to_dict() for r in records]

    # -----------------------------------------------------------------------
    # Detection Run Operations
    # -----------------------------------------------------------------------
    def save_run(
        self,
        dataset_name: str,
        metric_name: str,
        detection_result: Any,
        impact_report: Any | None = None,
        dataset_id: int | None = None,
    ) -> int:
        """
        Persist a detection run along with its change events and optional impact report.
        """
        with self.session_factory() as session:
            # If dataset_id not passed, find or create dataset record
            if dataset_id is None:
                ds = session.query(DatasetModel).filter_by(name=dataset_name).first()
                if ds:
                    dataset_id = ds.id

            # Create run record
            run = DetectionRunModel(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                metric_name=metric_name,
                algorithm_name=getattr(detection_result, "algorithm_name", "unknown"),
                params=json.dumps(getattr(detection_result, "params", {})),
                total_events=len(getattr(detection_result, "events", [])),
                execution_time_ms=getattr(detection_result, "execution_time_ms", 0.0),
            )
            session.add(run)
            session.flush()  # Populates run.id

            # Persist individual change events
            for ev in getattr(detection_result, "events", []):
                ts = getattr(ev, "timestamp", None)
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else (str(ts) if ts else None)
                event_model = ChangeEventModel(
                    run_id=run.id,
                    observation_index=getattr(ev, "index", 0),
                    event_timestamp=ts_str,
                    direction=getattr(ev, "direction", "unknown"),
                    score=float(getattr(ev, "score", 0.0)),
                    details=json.dumps(getattr(ev, "metadata", getattr(ev, "details", {}))),
                )
                session.add(event_model)

            # Persist optional impact report
            if impact_report is not None:
                self._attach_impact_report(session, run.id, metric_name, impact_report)

            session.commit()
            return run.id

    def _attach_impact_report(
        self,
        session: Session,
        run_id: int,
        metric_name: str,
        impact: Any,
    ) -> None:
        """Helper to serialize MetricImpact or ChangeImpactReport into the database."""
        # Case 1: ChangeImpactReport with multiple metrics
        if hasattr(impact, "metrics"):
            for m_name, m_impact in impact.metrics.items():
                if m_name == metric_name or len(impact.metrics) == 1:
                    self._create_impact_row(session, run_id, impact.change_index, m_impact)
                    return
            # If matching metric not found, store the first metric
            if impact.metrics:
                first_key = next(iter(impact.metrics))
                self._create_impact_row(session, run_id, impact.change_index, impact.metrics[first_key])
            return

        # Case 2: Direct MetricImpact or dictionary
        change_idx = impact.get("change_index", 0) if isinstance(impact, dict) else getattr(impact, "change_index", 0)
        self._create_impact_row(session, run_id, change_idx, impact)

    def _create_impact_row(
        self,
        session: Session,
        run_id: int,
        change_index: int,
        metric_impact: Any,
    ) -> None:
        """Helper to create an ImpactReportModel record from either MetricImpact object or dict."""
        def _get(key: str, default: Any = None) -> Any:
            if isinstance(metric_impact, dict):
                return metric_impact.get(key, default)
            return getattr(metric_impact, key, default)

        b_stats = _get("before_stats") or _get("pre_stats")
        a_stats = _get("after_stats") or _get("post_stats")

        pre_m = _get("pre_mean")
        if pre_m is None:
            pre_m = getattr(b_stats, "mean", 0.0) if b_stats else 0.0

        post_m = _get("post_mean")
        if post_m is None:
            post_m = getattr(a_stats, "mean", 0.0) if a_stats else 0.0

        report_model = ImpactReportModel(
            run_id=run_id,
            change_index=change_index,
            severity=_get("severity", "low"),
            cohen_d=float(_get("cohen_d", 0.0)),
            effect_size_category=_get("effect_size_category", "negligible"),
            p_value_welch=_get("p_value") or _get("p_value_welch"),
            p_value_mann_whitney=_get("mann_whitney_p_value") or _get("p_value_mann_whitney"),
            pre_mean=float(pre_m),
            post_mean=float(post_m),
            absolute_change=float(_get("absolute_change", 0.0)),
            percent_change=_get("percent_change"),
            pre_stats=json.dumps(b_stats.to_dict()) if hasattr(b_stats, "to_dict") else (json.dumps(b_stats) if isinstance(b_stats, dict) else None),
            post_stats=json.dumps(a_stats.to_dict()) if hasattr(a_stats, "to_dict") else (json.dumps(a_stats) if isinstance(a_stats, dict) else None),
            summary_sentence=_get("summary_sentence"),
        )
        session.add(report_model)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Load complete details for a detection run including events and impact reports."""
        with self.session_factory() as session:
            run = session.get(DetectionRunModel, run_id)
            if not run:
                return None

            result = run.to_dict()
            result["events"] = [e.to_dict() for e in run.events]
            reports = [r.to_dict() for r in run.impact_reports]
            result["impact_reports"] = reports
            result["impact_report"] = reports[0] if reports else None
            return result

    def list_runs(
        self,
        dataset_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List historical detection runs sorted chronologically descending."""
        with self.session_factory() as session:
            query = session.query(DetectionRunModel)
            if dataset_name:
                query = query.filter_by(dataset_name=dataset_name)

            runs = query.order_by(desc(DetectionRunModel.created_at)).limit(limit).all()
            return [r.to_dict() for r in runs]

    def delete_run(self, run_id: int) -> bool:
        """Delete a detection run and cascade to its events and impact reports."""
        with self.session_factory() as session:
            run = session.get(DetectionRunModel, run_id)
            if not run:
                return False
            session.delete(run)
            session.commit()
            return True

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset and cascade to its associated detection runs."""
        with self.session_factory() as session:
            dataset = session.get(DatasetModel, dataset_id)
            if not dataset:
                return False
            session.delete(dataset)
            session.commit()
            return True
