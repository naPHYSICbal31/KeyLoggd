"""
denoise.py

Low-pass filters a dwell/flight signal in the frequency domain and
reconstructs a cleaned time-domain version via the Inverse FFT.

Pipeline: x[n] --FFT--> X[k] --zero out high bins--> X'[k] --IFFT--> x_clean[n]

Why (for the report): human typing has natural run-to-run jitter (small
random timing noise on top of the "true" underlying rhythm). High-frequency
content in the spectrum largely corresponds to this jitter rather than to
the stable, identifying part of someone's typing rhythm. Suppressing it
before matching/classification should make templates more stable across
repetitions.
"""

import numpy as np

from fft_features import FFT_LEN, zero_pad


def low_pass_denoise(x: np.ndarray, cutoff_ratio: float = 0.35, n: int = FFT_LEN) -> np.ndarray:
    """
    x: 1-D real signal (dwell or flight array)
    cutoff_ratio: fraction of the (half) spectrum to KEEP, in [0, 1].
                  e.g. 0.35 keeps the lowest 35% of frequency bins and zeros
                  the rest, symmetric around DC to keep the signal real.
    n: FFT length (signal is zero-padded/truncated to this first)

    Returns the reconstructed, denoised time-domain signal, length n.
    """
    x_padded = zero_pad(x, n)
    X = np.fft.fft(x_padded)

    half = n // 2
    cutoff_bin = max(1, int(half * cutoff_ratio))

    X_filtered = X.copy()
    # zero out everything above cutoff_bin, on both sides of the spectrum,
    # so the result of the inverse FFT stays real-valued (conjugate symmetry)
    X_filtered[cutoff_bin : n - cutoff_bin] = 0.0

    x_clean = np.fft.ifft(X_filtered)

    # the imaginary part should be ~0 (numerical noise only) since we
    # preserved conjugate symmetry -- take the real part for the final signal
    return np.real(x_clean)


if __name__ == "__main__":
    # quick smoke test + a peek at how much jitter got removed
    from signal_construction import sample_to_signal
    from synthetic_data_generator import generate_sample, make_user_profile

    rng = np.random.default_rng(2)
    profile = make_user_profile(rng, "the quick brown fox")
    sample = generate_sample(rng, "the quick brown fox", profile)
    sig = sample_to_signal(sample)

    clean_dwell = low_pass_denoise(sig.dwell)
    print("raw dwell (padded):  ", np.round(zero_pad(sig.dwell), 4))
    print("denoised dwell:      ", np.round(clean_dwell, 4))
