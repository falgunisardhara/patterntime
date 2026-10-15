from pathlib import Path

import numpy as np
import pandas as pd


def generate_noise_scenarios():
    """
    Generate realistic time-series datasets to test detector reliability:
    1. High Noise: High standard deviation (sigma=30ms instead of 10ms) with a +70ms shift at index 60.
    2. Transient Spikes: Isolated one-off spikes (500ms at index 40 and 80) with NO permanent shift.
    3. Gradual Drift: Slow ramp-up (memory leak) starting at index 40, rising +2ms per observation.
    """
    rng = np.random.default_rng(42)

    total_points = 120
    timestamps = pd.date_range(
        start="2026-01-04 09:00:00",
        periods=total_points,
        freq="5min",
    )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Scenario 1: High Noise + Shift
    # Normal variance is large (std=30ms). A +70ms step occurs at index 60.
    # -------------------------------------------------------------
    high_noise_data = rng.normal(180, 30, total_points)
    high_noise_data[60:] += 70.0

    df_high_noise = pd.DataFrame({
        "timestamp": timestamps,
        "response_time_ms": np.maximum(high_noise_data, 10.0).round(2),
        "is_after_change": [1 if i >= 60 else 0 for i in range(total_points)],
    })
    high_noise_path = output_dir / "scenario_high_noise.csv"
    df_high_noise.to_csv(high_noise_path, index=False)
    print(f"Created {high_noise_path.name}: High noise (std=30ms) with +70ms shift at index 60")

    # -------------------------------------------------------------
    # Scenario 2: Transient Spikes (Outliers)
    # Baseline with std=10ms. Two isolated 500ms spikes at index 40 and 80.
    # No permanent shift occurs! A robust detector should NOT alert.
    # -------------------------------------------------------------
    spikes_data = rng.normal(180, 10, total_points)
    spikes_data[40] = 520.0  # One-off spike (e.g. garbage collection pause)
    spikes_data[80] = 490.0  # Second one-off spike

    df_spikes = pd.DataFrame({
        "timestamp": timestamps,
        "response_time_ms": spikes_data.round(2),
        "is_after_change": [0] * total_points,  # No permanent shift
    })
    spikes_path = output_dir / "scenario_transient_spikes.csv"
    df_spikes.to_csv(spikes_path, index=False)
    print(f"Created {spikes_path.name}: 2 isolated spikes (index 40 & 80), zero permanent changes")

    # -------------------------------------------------------------
    # Scenario 3: Gradual Drift (Memory Leak)
    # Drift starts at index 40 and climbs steadily until index 80 (+2ms per step).
    # -------------------------------------------------------------
    drift_data = rng.normal(180, 10, total_points)
    for i in range(40, 80):
        drift_data[i:] += 2.0  # climbs gradually by +2ms each point

    df_drift = pd.DataFrame({
        "timestamp": timestamps,
        "response_time_ms": drift_data.round(2),
        "is_after_change": [1 if i >= 40 else 0 for i in range(total_points)],
    })
    drift_path = output_dir / "scenario_gradual_drift.csv"
    df_drift.to_csv(drift_path, index=False)
    print(f"Created {drift_path.name}: Gradual drift starting at index 40 (+2ms/step)")


if __name__ == "__main__":
    generate_noise_scenarios()
