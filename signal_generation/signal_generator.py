from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def band_limited_noise(rng, n, fs, low, high, amplitude=1.0):
    """Generate band-limited noise using a Butterworth filter."""
    x = rng.normal(0, 1, n)

    nyquist = fs / 2
    low = max(low, 1.0) / nyquist
    high = min(high, nyquist * 0.98) / nyquist

    if not (0 < low < high < 1):
        raise ValueError("Invalid band limits for the selected sampling rate.")

    sos = butter(4, [low, high], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x)
    y /= np.std(y) + 1e-12

    return amplitude * y


def add_awgn(signal, snr_db, rng):
    """Add white Gaussian noise at a specified SNR."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def leak_signature(t, severity, rng):
    """
    Create a synthetic leak acoustic signature.

    Severity changes the overall acoustic energy and introduces modest
    changes in the spectral composition. A slowly varying envelope makes
    the signal less like a single artificial sinusoid.
    """
    severity_params = {
        "small":   {"amp": 0.35, "f1": 900,  "f2": 1450, "mod": 3.0},
        "medium":  {"amp": 0.65, "f1": 1150, "f2": 1750, "mod": 4.0},
        "large":   {"amp": 1.00, "f1": 1450, "f2": 2150, "mod": 5.0},
    }

    p = severity_params[severity]

    # Smoothly varying leak envelope
    envelope = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * p["mod"] * t))

    # Main leak components
    s = (
        np.sin(2 * np.pi * p["f1"] * t)
        + 0.55 * np.sin(2 * np.pi * p["f2"] * t + 0.7)
    )

    # Broadband turbulence-like component
    turbulence = band_limited_noise(
        rng, len(t), fs=1 / (t[1] - t[0]),
        low=500, high=3000, amplitude=0.30
    )

    return p["amp"] * envelope * (0.72 * s + 0.28 * turbulence)


def normal_flow_signal(t, rng, fs):
    """Create a baseline signal representing normal pipeline operation."""
    # Low-frequency flow/pump components
    pump = (
        0.20 * np.sin(2 * np.pi * 120 * t)
        + 0.12 * np.sin(2 * np.pi * 240 * t)
        + 0.08 * np.sin(2 * np.pi * 360 * t)
    )

    # Broadband flow turbulence
    flow_noise = band_limited_noise(
        rng, len(t), fs, low=150, high=1800, amplitude=0.18
    )

    return pump + flow_noise


def shift_signal(signal, delay_samples):
    """Shift a signal by an integer number of samples."""
    if delay_samples == 0:
        return signal.copy()

    shifted = np.zeros_like(signal)

    if delay_samples > 0:
        shifted[delay_samples:] = signal[:-delay_samples]
    else:
        d = abs(delay_samples)
        shifted[:-d] = signal[d:]

    return shifted


def generate_case(
    duration,
    fs,
    condition,
    severity,
    tdoa_ms,
    snr_db,
    seed,
):
    """
    Generate a two-sensor case.

    Sensor 1 receives the source signal first.
    Sensor 2 receives the same source after the configured TDOA.
    """
    rng = np.random.default_rng(seed)

    n = int(duration * fs)
    t = np.arange(n) / fs

    baseline = normal_flow_signal(t, rng, fs)

    if condition == "normal":
        source = baseline
    elif condition == "leak":
        source = baseline + leak_signature(t, severity, rng)
    else:
        raise ValueError("condition must be 'normal' or 'leak'")

    delay_samples = int(round(tdoa_ms * 1e-3 * fs))

    # Small amplitude difference models propagation/installation differences.
    sensor_1 = source
    sensor_2 = 0.90 * shift_signal(source, delay_samples)

    # Add independent measurement/environmental noise to each channel.
    sensor_1 = add_awgn(sensor_1, snr_db, rng)
    sensor_2 = add_awgn(sensor_2, snr_db, rng)

    # Normalize while retaining the relative relationship between channels.
    scale = max(np.max(np.abs(sensor_1)), np.max(np.abs(sensor_2)), 1e-12)
    sensor_1 /= scale
    sensor_2 /= scale

    return pd.DataFrame({
        "time_s": t,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2,
    })


def generate_dataset(
    output_dir,
    duration=2.0,
    fs=8000,
    tdoa_values=(0.75, 1.25, 2.00),
    snr_values=(5, 10, 20),
    repeats=5,
    seed=42,
):
    """Generate a labelled dataset covering multiple conditions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    case_id = 0

    for condition in ("normal", "leak"):
        severities = ("none",) if condition == "normal" else ("small", "medium", "large")

        for severity in severities:
            for snr_db in snr_values:
                for tdoa_ms in tdoa_values:
                    for repeat in range(repeats):
                        case_id += 1

                        df = generate_case(
                            duration=duration,
                            fs=fs,
                            condition=condition,
                            severity=severity if severity != "none" else "small",
                            tdoa_ms=tdoa_ms,
                            snr_db=snr_db,
                            seed=seed + case_id,
                        )

                        filename = (
                            f"case_{case_id:04d}_"
                            f"{condition}_{severity}_"
                            f"snr{snr_db}db_tdoa{tdoa_ms:.2f}ms.csv"
                        )

                        df.to_csv(output_dir / filename, index=False)

                        rows.append({
                            "case_id": case_id,
                            "filename": filename,
                            "condition": condition,
                            "severity": severity,
                            "snr_db": snr_db,
                            "tdoa_ms_ground_truth": tdoa_ms,
                            "sampling_rate_hz": fs,
                            "duration_s": duration,
                        })

    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)

    print(f"Generated {len(rows)} signal cases in: {output_dir}")
    print("Ground-truth TDOA values are stored in metadata.csv.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic two-channel pipeline acoustic signals."
    )

    parser.add_argument("--output", default="./data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Signal duration in seconds")
    parser.add_argument("--fs", type=int, default=8000,
                        help="Sampling frequency in Hz")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of repeats per scenario")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        duration=args.duration,
        fs=args.fs,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
