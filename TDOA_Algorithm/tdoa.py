"""
TDOA estimation using cross-correlation.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import correlate


def estimate_tdoa(signal_1, signal_2, fs):
    """Estimate relative delay between two sampled signals."""
    x = signal_1 - np.mean(signal_1)
    y = signal_2 - np.mean(signal_2)

    x /= np.std(x) + 1e-12
    y /= np.std(y) + 1e-12

    correlation = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-len(x) + 1, len(x))

    peak_index = np.argmax(np.abs(correlation))
    lag_samples = lags[peak_index]

    tdoa_seconds = lag_samples / fs
    peak_corr = correlation[peak_index] / len(x)

    return tdoa_seconds, float(peak_corr)


def estimate_location(tdoa_seconds, sensor_spacing, propagation_velocity):
    """
    Estimate source position between two sensors.

    Sensor 1 is at x=0 and Sensor 2 is at x=d.
    """
    d = sensor_spacing
    v = propagation_velocity

    x = (d - v * tdoa_seconds) / 2
    return float(np.clip(x, 0.0, d))


def process_file(input_file, fs, sensor_spacing, propagation_velocity,
                 ground_truth_tdoa=None):

    data = pd.read_csv(input_file)

    if {"sensor_1", "sensor_2"}.issubset(data.columns):
        signal_1 = data["sensor_1"].to_numpy(dtype=float)
        signal_2 = data["sensor_2"].to_numpy(dtype=float)
    elif {"sensor_1_filtered", "sensor_2_filtered"}.issubset(data.columns):
        signal_1 = data["sensor_1_filtered"].to_numpy(dtype=float)
        signal_2 = data["sensor_2_filtered"].to_numpy(dtype=float)
    else:
        raise ValueError("Input must contain raw or filtered sensor columns.")

    tdoa, peak_corr = estimate_tdoa(signal_1, signal_2, fs)

    location = estimate_location(
        tdoa, sensor_spacing, propagation_velocity
    )

    print(f"Estimated TDOA      : {tdoa * 1000:.4f} ms")
    print(f"Correlation peak    : {peak_corr:.4f}")
    print(f"Estimated location  : {location:.4f} m")

    if ground_truth_tdoa is not None:
        error = tdoa - ground_truth_tdoa
        print(f"Ground-truth TDOA   : {ground_truth_tdoa * 1000:.4f} ms")
        print(f"Absolute TDOA error : {abs(error) * 1000:.4f} ms")

    return tdoa, location


def main():
    parser = argparse.ArgumentParser(
        description="Estimate pipeline leak TDOA using cross-correlation."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--fs", type=int, default=8000)
    parser.add_argument("--sensor-spacing", type=float, default=2.0)
    parser.add_argument("--velocity", type=float, default=1000.0)
    parser.add_argument("--ground-truth-tdoa-ms", type=float, default=None)

    args = parser.parse_args()

    process_file(
        args.input,
        args.fs,
        args.sensor_spacing,
        args.velocity,
        None if args.ground_truth_tdoa_ms is None
        else args.ground_truth_tdoa_ms * 1e-3,
    )


if __name__ == "__main__":
    main()
