"""
day1_integration_test.py

End-to-end smoke test for Day 1:
  synthetic data -> signal construction -> FFT features -> inverse-FFT denoise

Also produces a quick plot comparing the dwell-time magnitude spectrum across
several synthetic users, as an early sanity check that different users
produce visually distinguishable spectra (the core assumption the whole
project rests on).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from denoise import low_pass_denoise
from fft_features import compute_spectrum, feature_vector, spectral_features
from signal_construction import load_user_signals

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic")
OUT_PLOT = os.path.join(os.path.dirname(__file__), "day1_spectra_comparison.png")


def main():
    user_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    assert user_files, f"No synthetic user files found in {DATA_DIR}. Run synthetic_data_generator.py first."

    plt.figure(figsize=(9, 5))

    for uf in user_files:
        with open(os.path.join(DATA_DIR, uf)) as f:
            record = json.load(f)

        signals = load_user_signals(record)

        # average the dwell-time magnitude spectrum across this user's samples
        mags = []
        feats = []
        for sig in signals:
            clean_dwell = low_pass_denoise(sig.dwell)
            spec = compute_spectrum(clean_dwell)
            sf = spectral_features(spec)
            mags.append(sf["magnitude"])
            feats.append(feature_vector(sig.dwell, sig.flight))

        mean_mag = np.mean(mags, axis=0)
        feats = np.array(feats)

        print(f"{record['user_id']}: {len(signals)} samples, "
              f"feature vector mean/std across samples:")
        print("  mean:", np.round(feats.mean(axis=0), 3))
        print("  std :", np.round(feats.std(axis=0), 3))

        plt.plot(mean_mag, label=record["user_id"])

    plt.title("Dwell-time FFT magnitude spectrum (denoised), averaged per user")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=130)
    print(f"\nSaved comparison plot to: {OUT_PLOT}")


if __name__ == "__main__":
    main()
