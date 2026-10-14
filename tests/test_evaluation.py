
from pathlib import Path
import sys

# Allow tests to import the project's current Python modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "patterntime"))

from evaluate_detector import match_events, calculate_metrics


def test_exact_event_match():
    matches, fp, fn = match_events([50], [50], tolerance=2)

    assert matches == [(50, 50)]
    assert fp == []
    assert fn == []


def test_event_within_tolerance():
    matches, fp, fn = match_events([52], [50], tolerance=2)

    assert matches == [(50, 52)]
    assert fp == []
    assert fn == []


def test_false_positive_and_false_negative():
    matches, fp, fn = match_events([10, 40], [12, 60], tolerance=2)

    assert matches == [(12, 10)]
    assert fp == [40]
    assert fn == [60]


def test_perfect_detection_metrics():
    metrics = calculate_metrics(
        matches=[(50, 50)],
        false_positives=[],
        false_negatives=[],
        true_change_count=1,
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mean_absolute_error_minutes"] == 0.0


def test_missed_event_metrics():
    metrics = calculate_metrics(
        matches=[],
        false_positives=[],
        false_negatives=[50],
        true_change_count=1,
    )

    assert metrics["true_positives"] == 0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["precision"] is None
    assert metrics["mean_absolute_error_minutes"] is None


def test_no_changes_and_no_detections():
    metrics = calculate_metrics(
        matches=[],
        false_positives=[],
        false_negatives=[],
        true_change_count=0,
    )

    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
