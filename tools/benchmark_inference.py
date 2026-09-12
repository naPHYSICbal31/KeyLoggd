"""
benchmark_inference.py

Measures the inference pipeline, so the accuracy claims in the pipeline
docstrings can be re-checked rather than trusted.

    python -m tools.benchmark_inference               # the full comparison
    python -m tools.benchmark_inference --real-only   # just data/synthetic

Why a fixture rather than the real captures
-------------------------------------------
The enrolled set is two people. An EER computed from a dozen samples moves in
8% steps and cannot resolve a change worth 3%, so a benchmark that only reads
data/synthetic/ cannot tell an improvement from noise. The fixture here gives
eight simulated users typing six different sentences each, which is enough to
measure -- and it deliberately reproduces the two things real keyboards do
that the plain synthetic generator does not:

  * Key rollover. A fast typist presses the next key before releasing the
    last, so flight times go negative -- 18% of one real user's flights here
    and 7% of the other's. The generator's max(0.01, ...) can never produce
    one, and a pipeline tuned only against it will mishandle the real thing.
  * Think-pauses. A ~1s gap mid-sentence, which dominates any feature built
    on a mean or a total.

Both are given a per-user rate, because how often someone rolls keys is part
of who they are. Uniform noise would only blur the users together; a
per-user habit is something a feature can legitimately find.

The real captures are still reported alongside, as a sanity check that the
fixture has not drifted away from them.

Protocol
--------
Leave-one-phrase-out for identification: templates are built from a user's
samples of the *other* sentences, then tested on the held-out sentence. This
is the honest free-text protocol -- leave-one-sample-out lets a sample be
scored against its own siblings from the same sentence, which flatters any
feature that encodes the sentence rather than the typist.

Verification EER is leave-one-sample-out against per-user templates, the
standard keystroke-biometrics measure.
"""

import argparse
import json
import os
import sys

import numpy as np

from keyloggd.paths import SYNTHETIC_DIR
from keyloggd.pipeline.classifier import (ZScoreScaler, build_templates,
                                          sample_distances, template_distance)
from keyloggd.pipeline.fft_features import feature_vector as spectral_vector
from keyloggd.pipeline.signal_construction import sample_to_signal
from keyloggd.pipeline.timing_features import feature_vector as timing_vector
from tools.synthetic_data_generator import generate_sample, make_user_profile

PHRASES = [
    "the quick brown fox",
    "sphinx of black quartz judge my vow",
    "pack my box with five dozen jugs",
    "how vexingly quick daft zebras jump",
    "we promptly judged antique ivory buckles",
    "jinxed wizards pluck ivy from the big quilt",
]


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def _add_keyboard_realism(sample, rng, rollover_p, pause_p, pause_scale):
    """Rebuild one event stream with this user's rollover and pause habits."""
    events = sample["events"]
    strokes = []
    for i, ev in enumerate(events):
        if ev["type"] != "down":
            continue
        up = next((e["t"] for e in events[i + 1:]
                   if e["type"] == "up" and e["key"] == ev["key"]), None)
        if up is not None:
            strokes.append((ev["key"], ev["t"], up))

    flat, t = [], 0.0
    for j, (key, down, up) in enumerate(strokes):
        hold = up - down
        flat.append({"key": key, "type": "down", "t": round(t, 6)})
        flat.append({"key": key, "type": "up", "t": round(t + hold, 6)})
        if j + 1 < len(strokes):
            gap = strokes[j + 1][1] - up
            if rng.random() < rollover_p:
                gap = -rng.uniform(0.005, 0.6) * hold   # next key goes down first
            elif rng.random() < pause_p:
                gap += rng.exponential(pause_scale)
            t += hold + gap
        else:
            t += hold

    flat.sort(key=lambda e: e["t"])
    return {"events": flat, "phrase": sample.get("phrase", "")}


def build_fixture(n_users=8, per_phrase=5, seed=23):
    """-> [(user_id, phrase, KeystrokeSignal), ...]"""
    rng = np.random.default_rng(seed)
    alphabet = "".join(sorted(set("".join(PHRASES))))
    rows = []
    for u in range(n_users):
        profile = make_user_profile(rng, alphabet)
        rollover_p = rng.uniform(0.03, 0.22)     # the range the real captures show
        pause_p = rng.uniform(0.01, 0.06)
        pause_scale = rng.uniform(0.15, 0.6)
        for phrase in PHRASES:
            for _ in range(per_phrase):
                s = generate_sample(rng, phrase, profile)
                s["phrase"] = phrase
                s = _add_keyboard_realism(s, rng, rollover_p, pause_p, pause_scale)
                rows.append((f"user_{u:02d}", phrase, sample_to_signal(s)))
    return rows


def load_real(data_dir=SYNTHETIC_DIR):
    rows = []
    if not os.path.isdir(data_dir):
        return rows
    for name in sorted(f for f in os.listdir(data_dir) if f.endswith(".json")):
        record = json.load(open(os.path.join(data_dir, name)))
        for s in record["samples"]:
            sig = sample_to_signal(s)
            if len(sig.dwell):
                rows.append((record["user_id"],
                             s.get("phrase", record.get("phrase", "")), sig))
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def featurize(rows, vector_fn):
    X = np.array([vector_fn(sig) for _, _, sig in rows], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array([u for u, _, _ in rows]), np.array([p for _, p, _ in rows])


def identification(X, y, groups, k=3, fuse=1, seed=0):
    """Closed-set accuracy, holding out a whole group (phrase) at a time.

    fuse > 1 pools that many same-user samples into one decision, the way a
    typing test hands over four or five at once.
    """
    rng = np.random.default_rng(seed)
    correct = total = 0
    for g in np.unique(groups):
        test, train = groups == g, groups != g
        if len(np.unique(y[train])) < 2:
            continue
        scaler = ZScoreScaler().fit(X[train])
        Xtr = scaler.transform(X[train])
        Xte = scaler.transform(X[test])
        yte = y[test]
        templates = build_templates(X[train], y[train], scaler)

        for user in np.unique(yte):
            idx = np.where(yte == user)[0]
            if fuse > 1:
                rng.shuffle(idx)
            for start in range(0, len(idx), fuse):
                block = idx[start:start + fuse]
                if len(block) < fuse:
                    break
                if fuse == 1:
                    # single sample: the kNN vote, as the app reports it
                    dists = sample_distances(Xtr, Xte[block[0]])
                    order = np.argsort(dists)[:k]
                    votes = {}
                    for i in order:
                        votes[y[train][i]] = (votes.get(y[train][i], 0.0)
                                              + 1.0 / (dists[i] + 1e-9))
                    pred = max(votes, key=votes.get)
                else:
                    pooled = {u: float(np.mean([template_distance(Xte[b], t)
                                                for b in block]))
                              for u, t in templates.items()}
                    pred = min(pooled, key=pooled.get)
                correct += int(pred == user)
                total += 1
    return correct / max(total, 1), total


def verification_eer(X, y):
    """Leave-one-sample-out genuine/impostor template scores -> EER."""
    users = np.unique(y)
    genuine, impostor = [], []
    for i in range(len(X)):
        train = np.ones(len(X), dtype=bool)
        train[i] = False
        if any(np.sum(y[train] == u) < 2 for u in users):
            continue                     # a user needs >=2 left for a spread
        scaler = ZScoreScaler().fit(X[train])
        query = scaler.transform(X[i:i + 1])[0]
        templates = build_templates(X[train], y[train], scaler)
        for u in users:
            d = template_distance(query, templates[u])
            (genuine if u == y[i] else impostor).append(d)

    genuine, impostor = np.array(genuine), np.array(impostor)
    if not len(genuine) or not len(impostor):
        return float("nan"), float("nan")
    grid = np.linspace(min(genuine.min(), impostor.min()),
                       max(genuine.max(), impostor.max()), 3000)
    best_gap, eer, at = 1e9, 1.0, 0.0
    for t in grid:
        far = float(np.mean(impostor <= t))
        frr = float(np.mean(genuine > t))
        if abs(far - frr) < best_gap:
            best_gap, eer, at = abs(far - frr), (far + frr) / 2, float(t)
    return eer, at


# ---------------------------------------------------------------------------

VECTORS = {
    "spectral (fft_features)": lambda sig: spectral_vector(sig.dwell, sig.flight),
    "timing  (timing_features)": lambda sig: timing_vector(sig.dwell, sig.flight, sig.keys),
}


def describe(rows, label):
    flight = np.concatenate([s.flight for _, _, s in rows if len(s.flight)])
    users = sorted({u for u, _, _ in rows})
    print(f"{label}: {len(rows)} samples, {len(users)} users, "
          f"{len({p for _, p, _ in rows})} distinct phrases, "
          f"{np.mean(flight < 0) * 100:.1f}% rollover")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--real-only", action="store_true",
                    help="skip the fixture; report only data/synthetic")
    ap.add_argument("--data-dir", default=SYNTHETIC_DIR)
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--per-phrase", type=int, default=5)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args(argv)

    real = load_real(args.data_dir)
    sets = []
    if not args.real_only:
        fixture = build_fixture(args.users, args.per_phrase, args.seed)
        describe(fixture, "cross-phrase fixture")
        sets.append(("fixture", fixture, True))
    if real:
        describe(real, "real captures     ")
        # one session per user, so leave-one-phrase-out is not available;
        # fall back to leave-one-sample-out and say so.
        sets.append(("real", real, False))
    if not sets:
        print(f"nothing to measure: no fixture and no files in {args.data_dir}")
        return 1
    print()

    for set_name, rows, by_phrase in sets:
        protocol = "leave-one-phrase-out" if by_phrase else "leave-one-sample-out"
        print(f"=== {set_name} ({protocol}) ===")
        print(f"{'feature vector':27s} {'dim':>4s} {'ident':>7s} "
              f"{'fuse3':>7s} {'fuse5':>7s} {'EER':>7s}")
        for name, fn in VECTORS.items():
            X, y, phrases = featurize(rows, fn)
            groups = phrases if by_phrase else np.arange(len(y))
            a1, _ = identification(X, y, groups, fuse=1)
            a3, n3 = identification(X, y, groups, fuse=3)
            a5, n5 = identification(X, y, groups, fuse=5)
            eer, _ = verification_eer(X, y)
            f3 = f"{a3 * 100:6.1f}%" if n3 else "     -"
            f5 = f"{a5 * 100:6.1f}%" if n5 else "     -"
            print(f"{name:27s} {X.shape[1]:4d} {a1 * 100:6.1f}% "
                  f"{f3:>7s} {f5:>7s} {eer * 100:6.2f}%")
        print()

    print("ident = one sample, distance-weighted kNN;  fuseN = N samples pooled")
    print("into one verdict (identify_sample.fuse);  EER = verification equal")
    print("error rate against per-user templates. Lower EER is better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
