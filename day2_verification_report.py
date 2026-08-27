"""
day2_verification_report.py

Runs the full identification + verification evaluation from classifier.py
and produces two report-ready plots:

  1. day2_score_distributions.png - histograms of genuine vs. impostor
     distance scores. Good separation between these two distributions is
     the whole ballgame for a verification system; this plot is the most
     direct visual evidence of whether the FFT feature vector is doing its
     job.

  2. day2_roc_curve.png - False Accept Rate vs. False Reject Rate swept
     over the distance threshold, with the Equal Error Rate point marked.
     Standard plot for reporting biometric system performance.

Run after synthetic_data_generator.py (or with real captured data, once
enough users/samples have been collected with capture_tool.html -- see
build_dataset(data_dir=...) to point at a different data folder).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifier import (
    build_dataset,
    equal_error_rate,
    loocv_identification_accuracy,
    print_confusion_matrix,
    verification_scores,
)

OUT_DIR = os.path.dirname(__file__)


def plot_score_distributions(genuine, impostor, out_path):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, max(genuine.max(), impostor.max()), 40)
    plt.hist(genuine, bins=bins, alpha=0.6, label=f"Genuine (n={len(genuine)})", color="#1a7a1a")
    plt.hist(impostor, bins=bins, alpha=0.6, label=f"Impostor (n={len(impostor)})", color="#b22222")
    plt.xlabel("Distance to claimed user's template")
    plt.ylabel("Count")
    plt.title("Genuine vs. Impostor Score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_roc(genuine, impostor, eer, thresh, out_path, n_thresholds=500):
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    fars = []
    frrs = []
    for t in thresholds:
        fars.append(np.mean(impostor <= t))
        frrs.append(np.mean(genuine > t))

    plt.figure(figsize=(6, 6))
    plt.plot(fars, frrs, color="#1f5fa8", label="FAR vs FRR")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="EER line (FAR=FRR)")
    plt.scatter([eer], [eer], color="red", zorder=5, label=f"EER = {eer * 100:.2f}%")
    plt.xlabel("False Accept Rate")
    plt.ylabel("False Reject Rate")
    plt.title("Verification ROC (DET-style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    X, y, user_ids = build_dataset()
    print(f"Loaded {len(X)} samples across {len(user_ids)} users: {user_ids}\n")

    print("=== Identification (LOOCV kNN, k=3) ===")
    acc, confusion = loocv_identification_accuracy(X, y, k=3)
    print(f"Accuracy: {acc:.3f} ({int(round(acc * len(X)))}/{len(X)})\n")
    print_confusion_matrix(confusion, user_ids)

    print("\n=== Verification ===")
    genuine, impostor = verification_scores(X, y, user_ids)
    eer, thresh = equal_error_rate(genuine, impostor)
    print(f"Genuine scores:  mean={genuine.mean():.3f}  std={genuine.std():.3f}")
    print(f"Impostor scores: mean={impostor.mean():.3f}  std={impostor.std():.3f}")
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%  (threshold ~= {thresh:.3f})")

    dist_path = os.path.join(OUT_DIR, "day2_score_distributions.png")
    roc_path = os.path.join(OUT_DIR, "day2_roc_curve.png")
    plot_score_distributions(genuine, impostor, dist_path)
    plot_roc(genuine, impostor, eer, thresh, roc_path)
    print(f"\nSaved: {dist_path}")
    print(f"Saved: {roc_path}")


if __name__ == "__main__":
    main()
