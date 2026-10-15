from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class ChangeEvent:
    """Represents a single detected change point in a time series."""

    index: int
    timestamp: pd.Timestamp | None = None
    direction: str = "unknown"  # "up", "down", or "unknown"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ChangeEvent to a JSON-serializable dictionary."""
        return {
            "index": self.index,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "direction": self.direction,
            "score": round(float(self.score), 4),
            "metadata": {
                k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                for k, v in self.metadata.items()
            },
        }


@dataclass
class DetectionResult:
    """Represents the complete result of a detector execution."""

    events: list[ChangeEvent]
    algorithm_name: str
    params: dict[str, Any]
    execution_time_ms: float

    @property
    def indices(self) -> list[int]:
        """List of integer indices where changes were detected."""
        return [event.index for event in self.events]

    @property
    def timestamps(self) -> list[pd.Timestamp]:
        """List of timestamps where changes were detected (if available)."""
        return [event.timestamp for event in self.events if event.timestamp is not None]

    @property
    def count(self) -> int:
        """Total number of detected change events."""
        return len(self.events)

    @property
    def has_changes(self) -> bool:
        """True if one or more changes were detected."""
        return len(self.events) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert DetectionResult to a JSON-serializable dictionary."""
        return {
            "algorithm_name": self.algorithm_name,
            "total_changes": self.count,
            "indices": self.indices,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "params": self.params,
            "events": [event.to_dict() for event in self.events],
        }


# ---------------------------------------------------------------------------
# Base Detector Interface
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    """Abstract base class for all time-series change-point detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Dictionary of current algorithm parameters."""

    @abstractmethod
    def _detect(self, series: pd.Series) -> list[ChangeEvent]:
        """Subclass implementation of the core detection algorithm."""

    def detect(
        self,
        series: pd.Series,
        timestamps: pd.Series | None = None,
    ) -> DetectionResult:
        """
        Execute change-point detection on a numeric time series.

        Parameters
        ----------
        series : pd.Series
            Numeric values representing the time series.
        timestamps : pd.Series, optional
            Corresponding timestamps for each observation.

        Returns
        -------
        DetectionResult
            Structured result containing detected events, timestamps, and metadata.
        """
        if series.empty:
            raise ValueError("Input series cannot be empty.")

        if series.isna().any():
            raise ValueError("Input series must not contain NaN or missing values.")

        if timestamps is not None:
            if len(timestamps) != len(series):
                raise ValueError(
                    f"Timestamps length ({len(timestamps)}) does not match series length ({len(series)})."
                )

        start_time = time.perf_counter()
        raw_events = self._detect(series)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Enrich events with timestamps if available
        if timestamps is not None:
            # Ensure timestamps are parsed
            ts_series = pd.to_datetime(timestamps).reset_index(drop=True)
            for event in raw_events:
                if 0 <= event.index < len(ts_series):
                    event.timestamp = ts_series.iloc[event.index]

        return DetectionResult(
            events=raw_events,
            algorithm_name=self.name,
            params=self.params,
            execution_time_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Concrete Detector Implementations
# ---------------------------------------------------------------------------
class SlidingWindowDetector(BaseDetector):
    """
    Detect changes by comparing adjacent sliding windows.
    Groups consecutive detections to isolate the peak change boundary.
    """

    def __init__(
        self,
        window_size: int = 12,
        threshold: float = 100.0,
        group_max_gap: int = 1,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if threshold < 0:
            raise ValueError("threshold cannot be negative.")
        if group_max_gap < 0:
            raise ValueError("group_max_gap cannot be negative.")

        self.window_size = window_size
        self.threshold = threshold
        self.group_max_gap = group_max_gap

    @property
    def name(self) -> str:
        return "Sliding Window"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "threshold": self.threshold,
            "group_max_gap": self.group_max_gap,
        }

    def _detect(self, series: pd.Series) -> list[ChangeEvent]:
        if len(series) < self.window_size * 2:
            return []

        candidates: list[dict[str, Any]] = []

        for index in range(self.window_size, len(series) - self.window_size + 1):
            earlier = series.iloc[index - self.window_size : index]
            recent = series.iloc[index : index + self.window_size]

            earlier_mean = float(earlier.mean())
            recent_mean = float(recent.mean())
            diff = recent_mean - earlier_mean

            if abs(diff) >= self.threshold:
                direction = "up" if diff > 0 else "down"
                candidates.append({
                    "index": index,
                    "difference": diff,
                    "direction": direction,
                    "earlier_mean": earlier_mean,
                    "recent_mean": recent_mean,
                })

        if not candidates:
            return []

        # Group adjacent candidates
        groups: list[list[dict[str, Any]]] = []
        current_group = [candidates[0]]

        for candidate in candidates[1:]:
            prev_index = current_group[-1]["index"]
            if candidate["index"] - prev_index <= self.group_max_gap:
                current_group.append(candidate)
            else:
                groups.append(current_group)
                current_group = [candidate]
        groups.append(current_group)

        # Select strongest candidate per group
        events: list[ChangeEvent] = []
        for group in groups:
            strongest = max(group, key=lambda item: abs(item["difference"]))
            events.append(
                ChangeEvent(
                    index=strongest["index"],
                    direction=strongest["direction"],
                    score=abs(strongest["difference"]),
                    metadata={
                        "difference": strongest["difference"],
                        "earlier_mean": strongest["earlier_mean"],
                        "recent_mean": strongest["recent_mean"],
                    },
                )
            )

        return events


class CusumDetector(BaseDetector):
    """
    Cumulative Sum (CUSUM) detector for tracking small, sustained shifts
    relative to a calibrated baseline. Supports optional Z-score clipping
    (Robust CUSUM) to suppress transient outliers.
    """

    def __init__(
        self,
        baseline_window: int = 20,
        drift: float = 0.5,
        threshold: float = 5.0,
        clip_z_score: float | None = None,
    ) -> None:
        if baseline_window < 2:
            raise ValueError("baseline_window must be at least 2.")
        if drift < 0:
            raise ValueError("drift cannot be negative.")
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero.")
        if clip_z_score is not None and clip_z_score <= 0:
            raise ValueError("clip_z_score must be greater than zero.")

        self.baseline_window = baseline_window
        self.drift = drift
        self.threshold = threshold
        self.clip_z_score = clip_z_score

    @property
    def name(self) -> str:
        return "Robust CUSUM" if self.clip_z_score is not None else "CUSUM"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "baseline_window": self.baseline_window,
            "drift": self.drift,
            "threshold": self.threshold,
            "clip_z_score": self.clip_z_score,
        }

    def _detect(self, series: pd.Series) -> list[ChangeEvent]:
        if len(series) < self.baseline_window:
            return []

        baseline = series.iloc[: self.baseline_window]
        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std(ddof=1))

        if baseline_std == 0:
            return []

        positive_sum = 0.0
        negative_sum = 0.0
        events: list[ChangeEvent] = []
        index = self.baseline_window

        while index < len(series):
            value = float(series.iloc[index])
            z_score = (value - baseline_mean) / baseline_std

            effective_z = z_score
            if self.clip_z_score is not None:
                effective_z = max(-self.clip_z_score, min(self.clip_z_score, z_score))

            positive_sum = max(0.0, positive_sum + effective_z - self.drift)
            negative_sum = min(0.0, negative_sum + effective_z + self.drift)

            direction = None
            score = 0.0

            if positive_sum >= self.threshold:
                direction = "up"
                score = positive_sum
            elif negative_sum <= -self.threshold:
                direction = "down"
                score = abs(negative_sum)

            if direction is not None:
                events.append(
                    ChangeEvent(
                        index=index,
                        direction=direction,
                        score=score,
                        metadata={
                            "positive_sum": positive_sum,
                            "negative_sum": negative_sum,
                            "baseline_mean": baseline_mean,
                            "baseline_std": baseline_std,
                            "z_score": z_score,
                        },
                    )
                )

                # Re-establish the baseline after the detected shift
                start = index + 1
                end = start + self.baseline_window

                if end > len(series):
                    break

                new_baseline = series.iloc[start:end]
                new_std = float(new_baseline.std(ddof=1))

                if not np.isfinite(new_std) or new_std <= 0:
                    break

                baseline_mean = float(new_baseline.mean())
                baseline_std = new_std
                positive_sum = 0.0
                negative_sum = 0.0
                index = end
                continue

            index += 1

        return events


class RobustCusumDetector(CusumDetector):
    """
    Pre-configured Robust CUSUM with Winsorization (clip_z_score=4.0)
    to ignore transient spikes while reliably catching sustained shifts.
    """

    def __init__(
        self,
        baseline_window: int = 20,
        drift: float = 0.5,
        threshold: float = 5.0,
        clip_z_score: float = 4.0,
    ) -> None:
        super().__init__(
            baseline_window=baseline_window,
            drift=drift,
            threshold=threshold,
            clip_z_score=clip_z_score,
        )


class EWMADetector(BaseDetector):
    """
    Exponentially Weighted Moving Average (EWMA) detector with dynamic control limits.
    Excels at detecting gradual drift, slow memory leaks, and subtle linear trends.
    Supports optional Z-score clipping to suppress isolated transient spikes.
    """

    def __init__(
        self,
        baseline_window: int = 20,
        lambda_: float = 0.2,
        control_limit: float = 3.0,
        clip_z_score: float | None = 4.0,
    ) -> None:
        if baseline_window < 2:
            raise ValueError("baseline_window must be at least 2.")
        if not (0.0 < lambda_ <= 1.0):
            raise ValueError("lambda_ (smoothing weight) must be in the range (0, 1].")
        if control_limit <= 0:
            raise ValueError("control_limit must be greater than zero.")
        if clip_z_score is not None and clip_z_score <= 0:
            raise ValueError("clip_z_score must be greater than zero.")

        self.baseline_window = baseline_window
        self.lambda_ = lambda_
        self.control_limit = control_limit
        self.clip_z_score = clip_z_score

    @property
    def name(self) -> str:
        return "EWMA"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "baseline_window": self.baseline_window,
            "lambda_": self.lambda_,
            "control_limit": self.control_limit,
            "clip_z_score": self.clip_z_score,
        }

    def _detect(self, series: pd.Series) -> list[ChangeEvent]:
        if len(series) < self.baseline_window:
            return []

        baseline = series.iloc[: self.baseline_window]
        mu = float(baseline.mean())
        sigma = float(baseline.std(ddof=1))

        if sigma == 0:
            return []

        z = mu
        events: list[ChangeEvent] = []
        step = 0

        for index in range(self.baseline_window, len(series)):
            step += 1
            x = float(series.iloc[index])

            # Optional outlier clipping (Winsorization)
            if self.clip_z_score is not None:
                z_val = (x - mu) / sigma
                clipped_z = max(-self.clip_z_score, min(self.clip_z_score, z_val))
                x = mu + clipped_z * sigma

            # Update EWMA statistic
            z = self.lambda_ * x + (1.0 - self.lambda_) * z

            # Compute exact dynamic control limit variance
            variance_factor = (self.lambda_ / (2.0 - self.lambda_)) * (
                1.0 - (1.0 - self.lambda_) ** (2 * step)
            )
            std_z = sigma * np.sqrt(variance_factor)

            ucl = mu + self.control_limit * std_z
            lcl = mu - self.control_limit * std_z

            direction = None
            score = 0.0

            if z > ucl:
                direction = "up"
                score = (z - ucl) / std_z if std_z > 0 else 1.0
            elif z < lcl:
                direction = "down"
                score = (lcl - z) / std_z if std_z > 0 else 1.0

            if direction is not None:
                events.append(
                    ChangeEvent(
                        index=index,
                        direction=direction,
                        score=score,
                        metadata={
                            "ewma_value": z,
                            "ucl": ucl,
                            "lcl": lcl,
                            "baseline_mean": mu,
                            "baseline_std": sigma,
                        },
                    )
                )

                # Reset baseline and EWMA tracking from post-change window
                if index + self.baseline_window <= len(series):
                    new_base = series.iloc[index : index + self.baseline_window]
                    mu = float(new_base.mean())
                    sigma = float(new_base.std(ddof=1))
                    z = mu
                    step = 0
                    if sigma == 0:
                        break
                else:
                    # Not enough points left to re-estimate baseline; stop
                    break

        return events


# ---------------------------------------------------------------------------
# Factory Registry
# ---------------------------------------------------------------------------
DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "sliding_window": SlidingWindowDetector,
    "cusum": CusumDetector,
    "robust_cusum": RobustCusumDetector,
    "ewma": EWMADetector,
}


def list_available_detectors() -> list[str]:
    """Return a list of all registered detector identifier names."""
    return sorted(DETECTOR_REGISTRY.keys())


def get_detector(name: str, **kwargs: Any) -> BaseDetector:
    """
    Instantiate a detector by its registered identifier name.

    Supported names:
    - "sliding_window"
    - "cusum"
    - "robust_cusum"
    - "ewma"
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector '{name}'. Available detectors: {list_available_detectors()}"
        )
    return DETECTOR_REGISTRY[key](**kwargs)
