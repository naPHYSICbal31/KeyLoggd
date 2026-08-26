"""
fft_features.py

Turns a dwell/flight discrete-time signal into a frequency-domain
representation via FFT, and extracts a compact feature vector from the
resulting spectrum.

Why FFT here (for the report):
    - Raw dwell/flight vectors are highly sensitive to small timing jitter
      run-to-run (a single slow keystroke shifts everything).
    - The magnitude spectrum summarizes the *rhythm structure* of the typing
      pattern (how energy is distributed across "typing frequencies") in a
      way that's more robust to small, localized timing perturbations than
      comparing raw time-domain vectors directly.
    - Zero-padding to a fixed length N lets us compare users/samples with
      slightly different raw lengths (e.g. if a keystroke event was dropped)
      on a common frequency grid.
"""

import numpy as np

FFT_LEN = 32  # fixed FFT length; signals are zero-padded/truncated to this


def zero_pad(x: np.ndarray, n: int = FFT_LEN) -> np.ndarray:
    """Zero-pad (or truncate) 1-D array x to length n."""
    x = np.asarray(x, dtype=float)
    if len(x) >= n:
        return x[:n]
    out = np.zeros(n, dtype=float)
    out[: len(x)] = x
    return out


def compute_spectrum(x: np.ndarray, n: int = FFT_LEN) -> np.ndarray:
    """
    Zero-pad x to length n and return the complex FFT (numpy.fft.fft).
    We keep the full complex spectrum here; magnitude/phase are derived by
    the caller depending on what's needed (features vs. reconstruction).
    """
    x_padded = zero_pad(x, n)
    return np.fft.fft(x_padded)


def spectral_features(spectrum: np.ndarray) -> dict:
    """
    Extract a small set of interpretable scalar features from a complex FFT
    spectrum. Only the first half (0 .. N/2) is used since the input signal
    is real-valued, so the spectrum is conjugate-symmetric and the second
    half carries no new information.
    """
    n = len(spectrum)
    half = n // 2 + 1
    mag = np.abs(spectrum[:half])
    freqs = np.arange(half)  # bin index acts as a proxy "frequency" axis

    total_energy = np.sum(mag ** 2) + 1e-12

    # spectral centroid: "center of mass" of the spectrum -> overall rhythm speed
    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-12)

    # energy in low vs high bins -> smoothness vs "jerkiness" of typing
    low_band = mag[: half // 3]
    mid_band = mag[half // 3 : 2 * half // 3]
    high_band = mag[2 * half // 3 :]

    low_energy = np.sum(low_band ** 2) / total_energy
    mid_energy = np.sum(mid_band ** 2) / total_energy
    high_energy = np.sum(high_band ** 2) / total_energy

    # dominant bin (excluding DC/bin 0, which is just the mean)
    dominant_bin = int(np.argmax(mag[1:]) + 1) if half > 1 else 0

    return {
        "magnitude": mag,  # full magnitude vector, useful for plotting
        "centroid": float(centroid),
        "low_energy_ratio": float(low_energy),
        "mid_energy_ratio": float(mid_energy),
        "high_energy_ratio": float(high_energy),
        "dominant_bin": dominant_bin,
        "total_energy": float(total_energy),
    }


def feature_vector(dwell: np.ndarray, flight: np.ndarray, n: int = FFT_LEN) -> np.ndarray:
    """
    Build one combined numeric feature vector for a typing sample, from both
    the dwell and flight signals. This is what gets fed to the classifier
    (kNN/SVM) later, and what gets compared distance-wise between a login
    attempt and stored per-user templates.
    """
    dwell_spec = compute_spectrum(dwell, n)
    flight_spec = compute_spectrum(flight, n)

    d_feat = spectral_features(dwell_spec)
    f_feat = spectral_features(flight_spec)

    vec = np.array(
        [
            d_feat["centroid"],
            d_feat["low_energy_ratio"],
            d_feat["mid_energy_ratio"],
            d_feat["high_energy_ratio"],
            d_feat["total_energy"],
            f_feat["centroid"],
            f_feat["low_energy_ratio"],
            f_feat["mid_energy_ratio"],
            f_feat["high_energy_ratio"],
            f_feat["total_energy"],
            # also include a few raw time-domain summary stats -- cheap and
            # genuinely useful signal, worth keeping alongside spectral feats
            float(np.mean(dwell)),
            float(np.std(dwell)),
            float(np.mean(flight)),
            float(np.std(flight)),
        ]
    )
    return vec


if __name__ == "__main__":
    # quick smoke test
    from signal_construction import sample_to_signal
    from synthetic_data_generator import generate_sample, make_user_profile

    rng = np.random.default_rng(1)
    profile = make_user_profile(rng, "the quick brown fox")
    sample = generate_sample(rng, "the quick brown fox", profile)
    sig = sample_to_signal(sample)

    vec = feature_vector(sig.dwell, sig.flight)
    print("feature vector shape:", vec.shape)
    print(np.round(vec, 4))
