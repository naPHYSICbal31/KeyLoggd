"""
identify_sample.py

"Guess who's typing" -- the actual deployment-style use case, as opposed to
classifier.py's LOOCV evaluation (which measures accuracy on data we already
have labels for). Here we:

  1. Enroll every user under data/synthetic/ (or --data-dir) using ALL of
     their samples (no held-out split -- this is the "real" enrollment step
     you'd do once per user).
  2. Take one or more brand-new, unlabeled samples (a JSON file in the same
     schema the capture tools write) and guess who typed each one.

For each unknown sample we report:
  - kNN vote across all enrolled samples (who do the k nearest neighbors say?)
  - distance to every enrolled user's template, closest first
  - a simple confidence signal: the gap between the closest and second
    closest template distance (a small gap means "could easily be either",
    a large gap means "confidently one user")

Usage:
    python -m keyloggd.pipeline.identify_sample path/to/unknown.json
    python -m keyloggd.pipeline.identify_sample unknown.json --k 3
"""

import argparse
import json
import os

import numpy as np

from keyloggd.paths import SYNTHETIC_DIR
from keyloggd.pipeline.classifier import (
    ZScoreScaler,
    build_dataset,
    build_templates,
    equal_error_rate,
    knn_predict,
    template_distance,
    verification_scores,
)
from keyloggd.pipeline.timing_features import signal_feature_vector
from keyloggd.pipeline.signal_construction import sample_to_signal

UNRECOGNIZED = "UNRECOGNIZED"

#: How far past the EER point to put the accept threshold.
#:
#: Chosen to minimise the half total error rate, HTER = (FAR + FRR) / 2:
#: the operating point that makes the fewest mistakes overall, counting a
#: stranger let in and an enrolled user turned away as equally bad.
#:
#: Measured on the cross-phrase benchmark (8 users), as a multiple of the
#: EER threshold, with the real enrolled set alongside:
#:
#:              benchmark                      real set (6 users)
#:             rejected  accepted  HTER       rejected  accepted  HTER
#:     x1.00      5.8%      6.1%   5.95%        18.8%     19.2%  18.96%
#:     x1.05      5.0%      7.3%   6.13%        12.5%     22.1%  17.29%
#:     x1.20      2.5%     11.7%   7.11%         8.3%     29.6%  18.96%
#:     x1.30      0.8%     14.5%   7.68%         6.2%     36.7%  21.46%
#:
#: The exact benchmark optimum is x1.06, so 1.05 sits on it, and it also
#: beats both 1.00 and 1.20 on the real set. The real set's own minimum
#: (x0.77) is not used: it rests on 48 genuine scores and lies at the edge
#: of the sweep, so it would be tuning to noise.
#:
#: This used to be 1.20, which deliberately traded false accepts for fewer
#: false rejects. If turning the enrolled user away matters more than
#: letting a stranger in, raise it again; that is a policy choice, not a
#: more accurate one.
#:
#: The threshold only decides whether the closest template is close *enough*,
#: never which one is closest, so moving it cannot change who a sample is
#: identified as -- only whether the answer is given at all.
ACCEPT_TOLERANCE = 1.05

DEFAULT_DATA_DIR = SYNTHETIC_DIR


def load_unknown_samples(path: str):
    """
    Reads an enrollment-schema JSON file:
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
        vecs.append(signal_feature_vector(sig))
    return np.array(vecs)


def compute_open_set_threshold(enrolled_X, enrolled_y, user_ids,
                               tolerance: float = ACCEPT_TOLERANCE):
    """
    Derive a distance threshold for "is this person enrolled at all?" using
    the same leave-one-out genuine/impostor scoring as classifier.py's
    verification eval, then picking the EER threshold and relaxing it by
    `tolerance`. Below this distance: treat as a plausible match. Above it:
    reject as unrecognized, regardless of which template happened to be
    closest.

    Returns (threshold, eer). The EER returned is the measured one at the
    equal-error point, not at the relaxed threshold -- it describes how well
    the enrolled set separates, which is a property of the data and does not
    change because the operating point moved. See ACCEPT_TOLERANCE.
    """
    genuine, impostor = verification_scores(enrolled_X, enrolled_y, user_ids)
    eer, threshold = equal_error_rate(genuine, impostor)
    return threshold * tolerance, eer


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

        dists = {u: template_distance(query, templates[u]) for u in user_ids}
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


def fuse(results, threshold=None):
    """One verdict from every sample of a run, instead of one vote per sample.

    A typing test emits four or five samples from a single sitting. Judging
    each alone throws away the fact that they are known to come from the same
    person: one unlucky sample -- a phone buzzing mid-sentence, a burst of
    unusually fast typing -- can sit far from its own template while its
    siblings sit right on it.

    Pooling the *distances* rather than voting on the per-sample answers is
    what makes that recoverable: a sample that was merely second-best still
    contributes evidence, where a vote discards it entirely. Averaging beat
    both median and min pooling on the benchmark, and takes cross-phrase
    identification from 93.3% on one sample to 97.9% on three and 100% on
    five.

    Returns the same shape as one entry of identify(), plus n_samples.
    """
    if not results:
        raise ValueError("no samples to fuse")

    users = sorted(dict(results[0]["ranked_distances"]))
    pooled = {
        u: float(np.mean([dict(r["ranked_distances"])[u] for r in results]))
        for u in users
    }
    ranked = sorted(pooled.items(), key=lambda kv: kv[1])

    closest_user, closest_dist = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else float("inf")
    accepted = threshold is None or closest_dist <= threshold

    return {
        "knn_vote": closest_user,
        "closest_template": closest_user,
        "closest_dist": closest_dist,
        "ranked_distances": ranked,
        "margin": second - closest_dist,
        "accepted": accepted,
        "decision": closest_user if accepted else UNRECOGNIZED,
        "n_samples": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="Guess who typed an unknown sample.")
    parser.add_argument("unknown_file", help="Path to a JSON file in the enrollment schema")
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
        "--tolerance",
        type=float,
        default=ACCEPT_TOLERANCE,
        help=f"How far past the equal-error point to put the accept "
             f"threshold (default {ACCEPT_TOLERANCE}). Above 1.0 is more "
             f"forgiving of the enrolled user, 1.0 is the strict EER point. "
             f"Ignored when --threshold is given.",
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
        threshold, eer = compute_open_set_threshold(
            enrolled_X, enrolled_y, user_ids, tolerance=args.tolerance)
        print(f"Auto distance threshold: {threshold:.3f}  "
              f"(EER point x{args.tolerance:g}; "
              f"enrolled-set EER = {eer * 100:.2f}%)\n")

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

    # Overall verdict: pooled distances across every sample, not a majority
    # vote of the per-sample answers -- see fuse().
    overall = fuse(results, threshold=threshold)
    print(f"=== Overall verdict across {overall['n_samples']} sample(s) ===")
    print(f"  Decision:            {overall['decision']}"
          + ("" if overall["accepted"] else "  (no template close enough)"))
    print(f"  Closest template:    {overall['closest_template']}  "
          f"(pooled dist={overall['closest_dist']:.3f}, "
          f"margin over 2nd: {overall['margin']:.3f})")
    print("  Pooled distances (closest first):")
    for u, d in overall["ranked_distances"]:
        print(f"    {u:12s} {d:.3f}")


if __name__ == "__main__":
    main()