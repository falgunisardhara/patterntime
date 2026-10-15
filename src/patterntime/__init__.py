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
