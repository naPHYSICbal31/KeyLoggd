"""
identify_sample.py

"Guess who's typing" -- the actual deployment-style use case, as opposed to
classifier.py's LOOCV evaluation (which measures accuracy on data we already
have labels for). Here we:

  1. Enroll every user under data/synthetic/ (or --data-dir) using ALL of
     their samples (no held-out split -- this is the "real" enrollment step
     you'd do once per user).
  2. Take one or more brand-new, unlabeled samples (a JSON file in the same
     format capture_tool.html downloads) and guess who typed each one.

For each unknown sample we report:
  - kNN vote across all enrolled samples (who do the k nearest neighbors say?)
  - distance to every enrolled user's template, closest first
  - a simple confidence signal: the gap between the closest and second
    closest template distance (a small gap means "could easily be either",
    a large gap means "confidently one user")

Usage:
    python3 identify_sample.py path/to/unknown.json
    python3 identify_sample.py path/to/unknown.json --data-dir data/synthetic --k 3
"""

import argparse
import json
import os

import numpy as np

from classifier import (
    ZScoreScaler,
    build_dataset,
    build_templates,
    equal_error_rate,
    knn_predict,
    verification_scores,
)
from fft_features import feature_vector
from signal_construction import sample_to_signal

UNRECOGNIZED = "UNRECOGNIZED"

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic")


def load_unknown_samples(path: str):
    """
    Reads a capture_tool.html-style JSON file:
        {"user_id": "...", "phrase": "...", "samples": [{"events": [...]}, ...]}
    The user_id field (if present) is ignored -- that's what we're trying to
    guess. Returns a list of feature vectors, one per sample/repetition.
    """
    with open(path) as f:
        record = json.load(f)

    vecs = []
    for s in record["samples"]:
        sig = sample_to_signal(s)
        if len(sig.dwell) == 0:
            continue  # malformed/empty sample, skip
        vecs.append(feature_vector(sig.dwell, sig.flight))
    return np.array(vecs)


def compute_open_set_threshold(enrolled_X, enrolled_y, user_ids):
    """
    Derive a distance threshold for "is this person enrolled at all?" using
    the same leave-one-out genuine/impostor scoring as classifier.py's
    verification eval, then picking the EER threshold. Below this distance:
    treat as a plausible match. Above it: reject as unrecognized, regardless
    of which template happened to be closest.
    """
    genuine, impostor = verification_scores(enrolled_X, enrolled_y, user_ids)
    eer, threshold = equal_error_rate(genuine, impostor)
    return threshold, eer


def identify(unknown_X, enrolled_X, enrolled_y, k=3, threshold=None):
    """
    Fit normalization + templates on the full enrolled set, then for each
    unknown sample report kNN vote + ranked template distances, plus an
    open-set accept/reject decision: if even the closest template is farther
    away than `threshold`, the sample is flagged UNRECOGNIZED instead of
    being forced onto whichever enrolled user happens to be least-bad.
    """
    scaler = ZScoreScaler().fit(enrolled_X)
    train_X = scaler.transform(enrolled_X)
    templates = build_templates(enrolled_X, enrolled_y, scaler)
    user_ids = sorted(templates.keys())

    results = []
    for query_raw in unknown_X:
        query = scaler.transform(query_raw.reshape(1, -1))[0]

        vote = knn_predict(train_X, enrolled_y, query, k=k)

        dists = {u: float(np.linalg.norm(query - templates[u])) for u in user_ids}
        ranked = sorted(dists.items(), key=lambda kv: kv[1])

        closest_user, closest_dist = ranked[0]
        second_dist = ranked[1][1] if len(ranked) > 1 else float("inf")
        margin = second_dist - closest_dist

        accepted = threshold is None or closest_dist <= threshold
        decision = vote if accepted else UNRECOGNIZED

        results.append(
            {
                "knn_vote": vote,
                "closest_template": closest_user,
                "closest_dist": closest_dist,
                "ranked_distances": ranked,
                "margin": margin,
                "accepted": accepted,
                "decision": decision,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Guess who typed an unknown sample.")
    parser.add_argument("unknown_file", help="Path to a capture_tool.html-style JSON file")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory of enrolled user JSON files")
    parser.add_argument("--k", type=int, default=3, help="k for kNN vote")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold for accept/reject. Default: auto-computed EER "
        "threshold from the enrolled set. Pass --no-threshold to disable "
        "open-set rejection entirely (always force a guess, old behavior).",
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Disable open-set rejection; always output the closest match, "
        "even for a likely stranger.",
    )
    args = parser.parse_args()

    enrolled_X, enrolled_y, user_ids = build_dataset(args.data_dir)
    print(f"Enrolled: {len(enrolled_X)} samples across {len(user_ids)} users: {user_ids}\n")

    if args.no_threshold:
        threshold = None
        print("Open-set rejection disabled (--no-threshold): will always guess an enrolled user.\n")
    elif args.threshold is not None:
        threshold = args.threshold
        print(f"Using manual distance threshold: {threshold:.3f}\n")
    else:
        threshold, eer = compute_open_set_threshold(enrolled_X, enrolled_y, user_ids)
        print(f"Auto distance threshold: {threshold:.3f}  (from enrolled-set EER = {eer * 100:.2f}%)\n")

    unknown_X = load_unknown_samples(args.unknown_file)
    print(f"Loaded {len(unknown_X)} unknown sample(s) from {args.unknown_file}\n")

    results = identify(unknown_X, enrolled_X, enrolled_y, k=args.k, threshold=threshold)

    for i, r in enumerate(results):
        print(f"--- Sample {i + 1} ---")
        print(f"  kNN vote (k={args.k}):      {r['knn_vote']}")
        print(f"  Closest template:      {r['closest_template']}  (dist={r['closest_dist']:.3f}, margin over 2nd place: {r['margin']:.3f})")
        print(f"  Decision:              {r['decision']}" + ("" if r["accepted"] else "  (closest match was still farther than threshold)"))
        print("  Ranked distances (closest first):")
        for u, d in r["ranked_distances"]:
            print(f"    {u:12s} {d:.3f}")
        print()

    # overall guess across all provided unknown samples (majority of per-sample decisions)
    decisions = [r["decision"] for r in results]
    labels, counts = np.unique(decisions, return_counts=True)
    overall = labels[np.argmax(counts)]
    print(f"Overall decision across all {len(results)} sample(s): {overall}")


if __name__ == "__main__":
    main()