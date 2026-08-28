"""
day3_cross_phrase_eval.py

Does this system still recognise you when you type a *different sentence*?

Day 1 and Day 2 measured fixed-text performance: everyone types the same
enrollment phrase, and the question is whether the rhythm identifies the
person. That is the easier half of keystroke biometrics. This script measures
the harder half -- free text -- and checks whether the feature set survives it.

Three protocols, all using the same enrollment/scoring code as the rest of the
project (classifier.ZScoreScaler / build_templates / knn_predict /
equal_error_rate):

    A  same-sentence      enroll and test on the same sentence (leave one
                          sample out). This is what Day 1/2 report.
    B  free text          enroll on a user's *other* sentences, test on a
                          sentence the system has never seen them type.
    C  length-matched     protocol B restricted to sentences of near-identical
                          length. Without this control a "pass" can just be
                          the classifier reading how long the text was.

Each protocol runs twice: once with the original zero-padded features
(mode="pad") and once with length-invariant resampled features
(mode="resample"), so the difference is attributable to that one change.

Run:
    python day3_cross_phrase_eval.py
"""

import collections
import os

import numpy as np

from classifier import ZScoreScaler, build_templates, equal_error_rate, knn_predict
from fft_features import feature_vector
from signal_construction import sample_to_signal
from synthetic_data_generator import MULTI_OUT_DIR, generate_multiphrase_dataset

# the phrase pool the capture tool offers, so the fixture matches the app
PHRASES = [
    "the quick brown fox",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "sphinx of black quartz judge my vow",
    "waltz nymph for quick jigs vex bud",
    "bright vixens jump dozy fowl quack",
    "jinxed wizards pluck ivy from the big quilt",
    "two driven jocks help fax my big quiz",
    "quick zephyrs blow vexing daft jim",
    "my ex pub quiz crowd finally won jackpot",
    "grumpy wizards make toxic brew for the jovial queen",
]

N_USERS = 8
N_SAMPLES_PER_PHRASE = 6
K = 3
LENGTH_BAND = (34, 37)   # sentences close enough in length to rule length out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def ensure_dataset():
    if not os.path.isdir(MULTI_OUT_DIR) or not os.listdir(MULTI_OUT_DIR):
        print(f"generating multi-sentence fixture in {MULTI_OUT_DIR} ...")
        generate_multiphrase_dataset(PHRASES, n_users=N_USERS,
                                     n_samples_per_phrase=N_SAMPLES_PER_PHRASE)
        print()
    return MULTI_OUT_DIR


def load_samples(data_dir):
    """[(user_id, phrase, dwell, flight)] for every sample in the fixture."""
    import json

    out = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(data_dir, name)) as f:
            record = json.load(f)
        for sample in record["samples"]:
            signal = sample_to_signal(sample)
            if len(signal.dwell) == 0:
                continue
            phrase = sample.get("phrase") or record.get("phrase", "")
            out.append((record["user_id"], phrase, signal.dwell, signal.flight))
    return out


def featurise(samples, mode):
    X = np.array([feature_vector(d, f, mode=mode) for _, _, d, f in samples])
    y = np.array([u for u, _, _, _ in samples])
    phrases = np.array([p for _, p, _, _ in samples])
    return X, y, phrases


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_split(train_X, train_y, test_X, test_y):
    """Enroll on the train half, then identify and verify the test half.

    Returns (nearest-template accuracy, kNN accuracy, EER). Normalisation and
    templates are fit on the training half only, so nothing about the held-out
    sentence leaks into enrollment.
    """
    scaler = ZScoreScaler().fit(train_X)
    train_Z = scaler.transform(train_X)
    templates = build_templates(train_X, train_y, scaler)
    users = sorted(templates)

    nearest_hits = knn_hits = 0
    genuine, impostor = [], []

    for x, truth in zip(test_X, test_y):
        z = scaler.transform(x.reshape(1, -1))[0]
        distances = {u: float(np.linalg.norm(z - templates[u])) for u in users}
        nearest = min(distances, key=distances.get)
        nearest_hits += nearest == truth
        knn_hits += knn_predict(train_Z, train_y, z, k=K) == truth
        for u, d in distances.items():
            (genuine if u == truth else impostor).append(d)

    n = max(len(test_X), 1)
    eer = float("nan")
    if genuine and impostor:
        eer, _ = equal_error_rate(np.array(genuine), np.array(impostor))
    return nearest_hits / n, knn_hits / n, eer


def protocol_same_sentence(X, y, phrases):
    """A: leave one sample out, enrolling on the same sentence only."""
    nearest, knn, eers = [], [], []
    for phrase in sorted(set(phrases)):
        mask = phrases == phrase
        Xp, yp = X[mask], y[mask]
        for i in range(len(Xp)):
            hold = np.zeros(len(Xp), dtype=bool)
            hold[i] = True
            a, b, _ = score_split(Xp[~hold], yp[~hold], Xp[hold], yp[hold])
            nearest.append(a)
            knn.append(b)
        # EER once per sentence over its own held-out scoring
        a, b, eer = score_split(Xp, yp, Xp, yp)
        eers.append(eer)
    return np.mean(nearest), np.mean(knn), np.nanmean(eers)


def protocol_leave_one_sentence_out(X, y, phrases, allowed=None):
    """B/C: enroll on every other sentence, test on the held-out one."""
    pool = sorted(set(phrases)) if allowed is None else sorted(allowed)
    keep = np.isin(phrases, pool)
    X, y, phrases = X[keep], y[keep], phrases[keep]

    nearest, knn, eers = [], [], []
    for phrase in pool:
        test = phrases == phrase
        if not test.any() or test.all():
            continue
        a, b, eer = score_split(X[~test], y[~test], X[test], y[test])
        nearest.append(a)
        knn.append(b)
        eers.append(eer)
    return np.mean(nearest), np.mean(knn), np.nanmean(eers)


def distance_table(X, y, phrases):
    """The three distances that explain the result better than accuracy does.

    All three are measured the same way -- one sample against a template built
    from six samples -- so they are comparable. (Comparing sample-to-sample
    distances against template distances would make the cross-sentence number
    look artificially small, because averaging six samples cancels noise the
    single sample still carries.)
    """
    Z = ZScoreScaler().fit(X).transform(X)
    groups = collections.defaultdict(list)
    for i, (u, p) in enumerate(zip(y, phrases)):
        groups[(u, p)].append(Z[i])

    templates = {key: np.mean(vs, axis=0) for key, vs in groups.items()}
    same_sentence, other_sentence, other_person = [], [], []

    for (user, phrase), vectors in groups.items():
        for vector in vectors:
            # own template for this sentence, with this sample left out
            rest = [v for v in vectors if v is not vector]
            if rest:
                same_sentence.append(
                    np.linalg.norm(vector - np.mean(rest, axis=0)))
            for (other_u, other_p), template in templates.items():
                if other_u == user and other_p != phrase:
                    other_sentence.append(np.linalg.norm(vector - template))
                elif other_u != user and other_p == phrase:
                    other_person.append(np.linalg.norm(vector - template))

    return (np.mean(same_sentence), np.mean(other_sentence),
            np.mean(other_person))


# indices into feature_vector's 14 numbers
SPECTRAL_FEATURES = list(range(0, 10))
TIMING_FEATURES = list(range(10, 14))


def ablation(samples, mode, band):
    """Which half of the feature vector actually carries identity across
    sentences: the ten FFT numbers, or the four plain timing statistics?"""
    X, y, phrases = featurise(samples, mode)
    out = {}
    for label, columns in (("all 14", None),
                           ("spectral only (10)", SPECTRAL_FEATURES),
                           ("timing only (4)", TIMING_FEATURES)):
        Xs = X if columns is None else X[:, columns]
        free = protocol_leave_one_sentence_out(Xs, y, phrases)
        matched = protocol_leave_one_sentence_out(Xs, y, phrases, allowed=band)
        out[label] = (free[0], matched[0])
    return out


REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "synthetic")


def real_data_check(data_dir=REAL_DATA_DIR):
    """The same held-out-sentence test on the real captures.

    Only users with more than one sentence can be tested this way, and today
    that is one person -- so this is an anecdote, not a measurement. It is
    reported because it is the only human evidence available, and because it
    shows the confound the synthetic fixture cannot: in the real set that
    person is the *only* one typing long sentences, so sentence length doubles
    as their name.
    """
    if not os.path.isdir(data_dir):
        return
    samples = load_samples(data_dir)
    by_user = collections.defaultdict(set)
    for user, phrase, _, _ in samples:
        by_user[user].add(phrase)
    multi = [u for u, ps in by_user.items() if len(ps) > 1]
    if not multi:
        return

    print(f"\n--- real captures in {os.path.basename(data_dir)}/ "
          f"(multi-sentence users: {', '.join(multi)}) ---")
    for mode in ("pad", "resample"):
        X, y, phrases = featurise(samples, mode)
        for user in multi:
            hits = shared_hits = shared_total = 0
            total = 0
            for phrase in sorted(by_user[user]):
                held = (y == user) & (phrases == phrase)
                if not held.any():
                    continue
                near, _, _ = score_split(X[~held], y[~held], X[held], y[held])
                hits += near * held.sum()
                total += held.sum()
                # sentences other users also type: length cannot help there
                if len({u for u, p in zip(y, phrases)
                        if p == phrase and u != user}) > 0:
                    shared_hits += near * held.sum()
                    shared_total += held.sum()
            line = (f"  {user:9s} [{mode:8s}] held-out sentence: "
                    f"{hits:.0f}/{total} correct")
            if shared_total:
                line += (f"   on sentences others also type: "
                         f"{shared_hits:.0f}/{shared_total}")
            print(line)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    data_dir = ensure_dataset()
    samples = load_samples(data_dir)
    users = sorted({u for u, _, _, _ in samples})
    phrase_set = sorted({p for _, p, _, _ in samples}, key=len)
    band = [p for p in phrase_set if LENGTH_BAND[0] <= len(p) <= LENGTH_BAND[1]]

    print(f"{len(samples)} samples | {len(users)} users | "
          f"{len(phrase_set)} sentences ({len(phrase_set[0])}-"
          f"{len(phrase_set[-1])} chars)")
    print(f"length-matched control uses {len(band)} sentences of "
          f"{LENGTH_BAND[0]}-{LENGTH_BAND[1]} chars\n")

    chance = 1.0 / len(users)
    results = {}
    for mode in ("pad", "resample"):
        X, y, phrases = featurise(samples, mode)
        results[mode] = {
            "A same-sentence": protocol_same_sentence(X, y, phrases),
            "B free text": protocol_leave_one_sentence_out(X, y, phrases),
            "C length-matched": protocol_leave_one_sentence_out(X, y, phrases,
                                                               allowed=band),
        }
        results[mode]["distances"] = distance_table(X, y, phrases)

    header = f"{'protocol':18s} {'nearest':>9s} {'kNN':>8s} {'EER':>8s}"
    for mode, label in (("pad", "zero-padded (current)"),
                        ("resample", "resampled (length-invariant)")):
        print(f"--- {label} ---")
        print(header)
        for name in ("A same-sentence", "B free text", "C length-matched"):
            near, knn, eer = results[mode][name]
            print(f"{name:18s} {near * 100:8.1f}% {knn * 100:7.1f}% "
                  f"{eer * 100:7.1f}%")
        same, other_s, other_p = results[mode]["distances"]
        print(f"  distance  same person/same sentence {same:5.2f}   "
              f"same person/other sentence {other_s:5.2f}   "
              f"other person/same sentence {other_p:5.2f}")
        verdict = ("sentence change costs more than identity"
                   if other_s > other_p else
                   "identity separates more than sentence does")
        print(f"  -> {verdict}\n")

    print(f"chance level with {len(users)} users: {chance * 100:.1f}%")
    pad_free = results["pad"]["B free text"][0]
    res_free = results["resample"]["B free text"][0]
    pad_ctrl = results["pad"]["C length-matched"][0]
    res_ctrl = results["resample"]["C length-matched"][0]
    print(f"free text:      {pad_free * 100:.1f}% -> {res_free * 100:.1f}% "
          f"after resampling")
    print(f"length-matched: {pad_ctrl * 100:.1f}% -> {res_ctrl * 100:.1f}%")

    print("\n--- which features carry identity across sentences? "
          "(nearest-template) ---")
    print(f"{'features':30s} {'free text':>11s} {'length-matched':>16s}")
    for mode in ("pad", "resample"):
        for label, (free, matched) in ablation(samples, mode, band).items():
            print(f"{label + ' [' + mode + ']':30s} {free * 100:10.1f}% "
                  f"{matched * 100:15.1f}%")

    real_data_check()

    print("\nRead this fixture with care: the generator models a person as a "
          "constant base speed plus a per-key bias, which is sentence-"
          "independent by construction, so the four timing features transfer "
          "perfectly. Real typists slow on unfamiliar words and pause at "
          "phrase boundaries, so treat these as an upper bound on free-text "
          "accuracy, and the fixed-text numbers from day1/day2 as the only "
          "measured-on-humans result so far.")


if __name__ == "__main__":
    main()
