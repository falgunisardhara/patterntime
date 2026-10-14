
import math

import pandas as pd


def detect_cusum(
    values: pd.Series,
    baseline_window: int = 20,
    drift: float = 0.5,
    threshold: float = 5.0,
) -> list[dict]:
    """
    Detect upward and downward shifts using standardized CUSUM.

    After detecting a shift, use the following baseline_window
    observations to establish a new baseline before continuing.
    The detection index represents when the shift was detected,
    not necessarily the exact point where the shift began.
    """
    if baseline_window < 2: # Estimates the normal mean and standard deviation
        raise ValueError("baseline_window must be at least 2.")
    if drift < 0:   # Controls how much standardized deviation is tolerated before evidence accumulates
        raise ValueError("drift cannot be negative.")
    if threshold <= 0:  # Determines how much accumulated evidence triggers a detection
        raise ValueError("threshold must be greater than zero.")
    if values.isna().any():
        raise ValueError("Input values must not contain missing values.")

    values = values.reset_index(drop=True).astype(float)

    if len(values) <= baseline_window:
        raise ValueError("Not enough observations for baseline and detection.")

    baseline = values.iloc[:baseline_window]
    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std(ddof=1))

    if not math.isfinite(baseline_std) or baseline_std <= 0:
        raise ValueError("Initial baseline must have non-zero variation.")

    detections = []
    positive_sum = 0.0
    negative_sum = 0.0
    index = baseline_window

    while index < len(values):
        value = float(values.iloc[index])
        z_score = (value - baseline_mean) / baseline_std

        positive_sum = max(0.0, positive_sum + z_score - drift)
        negative_sum = min(0.0, negative_sum + z_score + drift)

        direction = None

        if positive_sum >= threshold:
            direction = "up"
        elif negative_sum <= -threshold:
            direction = "down"

        if direction is not None:
            detections.append({
                "index": index,
                "direction": direction,
                "value": round(value, 2),
                "baseline_mean": round(baseline_mean, 2),
                "baseline_std": round(baseline_std, 2),
                "z_score": round(z_score, 2),
            })

            # Re-establish the baseline after the detected shift.
            start = index + 1
            end = start + baseline_window

            if end > len(values):
                break

            new_baseline = values.iloc[start:end]
            new_std = float(new_baseline.std(ddof=1))

            if not math.isfinite(new_std) or new_std <= 0:
                break

            baseline_mean = float(new_baseline.mean())
            baseline_std = new_std
            positive_sum = 0.0
            negative_sum = 0.0
            index = end
            continue

        index += 1

    return detections
