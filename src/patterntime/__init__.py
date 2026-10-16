"""
PatternTime: Interactive Time-Series Investigation Laboratory
"""

from patterntime.detect_changes import detect_changes, group_detections
from patterntime.detect_cusum import detect_cusum
from patterntime.detectors import (
    BaseDetector,
    ChangeEvent,
    CusumDetector,
    DetectionResult,
    EWMADetector,
    RobustCusumDetector,
    SlidingWindowDetector,
    get_detector,
    list_available_detectors,
)
from patterntime.impact import (
    ChangeImpactReport,
    MetricImpact,
    SegmentStats,
    analyze_all_changes,
    analyze_change_impact,
    analyze_metric_shift,
    calculate_cohen_d,
    classify_severity,
    compute_segment_stats,
)
from patterntime.validator import (
    DuplicateTimestampError,
    EmptyDataError,
    InsufficientDataError,
    InvalidTimestampError,
    MissingTimestampError,
    MissingValueError,
    NoNumericMetricsError,
    TimeSeriesValidationError,
    ValidationConfig,
    ValidationResult,
    ZeroVarianceError,
    load_and_validate_csv,
    validate_dataframe,
)

__all__ = [
    # Baseline functions
    "detect_changes",
    "group_detections",
    "detect_cusum",
    # Unified Detectors & Data Models
    "BaseDetector",
    "ChangeEvent",
    "DetectionResult",
    "SlidingWindowDetector",
    "CusumDetector",
    "RobustCusumDetector",
    "EWMADetector",
    "get_detector",
    "list_available_detectors",
    # Impact Analysis Engine
    "SegmentStats",
    "MetricImpact",
    "ChangeImpactReport",
    "compute_segment_stats",
    "calculate_cohen_d",
    "classify_severity",
    "analyze_metric_shift",
    "analyze_change_impact",
    "analyze_all_changes",
    # Validator module
    "load_and_validate_csv",
    "validate_dataframe",
    "ValidationConfig",
    "ValidationResult",
    "TimeSeriesValidationError",
    "EmptyDataError",
    "InsufficientDataError",
    "MissingTimestampError",
    "InvalidTimestampError",
    "DuplicateTimestampError",
    "MissingValueError",
    "ZeroVarianceError",
    "NoNumericMetricsError",
]
