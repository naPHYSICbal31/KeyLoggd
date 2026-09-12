"""
classifier.py

Day 2: turn per-sample feature vectors into (a) an identification system
("who is typing?") and (b) a verification system ("is this really user X?"),
both built on the feature vectors from timing_features.feature_vector.

Design notes (for the report):

- Feature scales differ wildly (ratios ~0-1, raw dwell/flight stats
  ~0.02-0.3, keys-per-second ~2-8), so every distance-based method here
  z-score normalizes features first (fit on the training/enrollment data
  only, to avoid leaking test-set statistics).

- Distances are Manhattan, not Euclidean, and a sample-to-template distance
  is additionally divided by that user's own per-feature spread. Both
  choices are measured, not assumed -- see sample_distances and
  template_distance for the reasoning and the numbers.

- Two complementary evaluations are implemented, mirroring how keystroke
  biometrics is actually evaluated in the literature:

    1. Identification (closed-set, 1-of-N): given a sample, which of the
       N enrolled users produced it? Evaluated with a k-nearest-neighbor
       classifier and leave-one-out cross-validation (LOOCV), reported as
       accuracy.

    2. Verification (1-to-1, open-set-style): given a sample and a claimed
       identity, is it a match? Each sample is scored against its claimed
       user's enrollment template (genuine score) and against every other
       user's template (impostor scores). Reported as an ROC-style sweep
       and Equal Error Rate (EER) -- the standard headline metric in
       biometrics, where False Accept Rate == False Reject Rate.

No external ML dependencies (sklearn etc.) are used -- everything is plain
numpy, since the whole point of the project is to build the pipeline from
first principles for the report.
"""

import json
import os

import numpy as np

from keyloggd.paths import SYNTHETIC_DIR
from keyloggd.pipeline.signal_construction import load_user_signals
from keyloggd.pipeline.timing_features import signal_feature_vector

DATA_DIR = SYNTHETIC_DIR


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str = DATA_DIR):
    """
    Load every user JSON file in data_dir and turn it into a flat feature
    matrix, ready for classification/verification.

    Returns:
        X:        (n_samples, n_features) float array
        y:        (n_samples,) array of user_id strings, one per row of X
        user_ids: sorted list of unique user_id strings
    """
    user_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    assert user_files, f"No user files found in {data_dir}"

    X_rows = []
    y_rows = []

    for uf in user_files:
        with open(os.path.join(data_dir, uf)) as f:
            record = json.load(f)

        signals = load_user_signals(record)
        for sig in signals:
            if len(sig.dwell) == 0:
                continue  # malformed sample: no usable keystrokes
            X_rows.append(signal_feature_vector(sig))
            y_rows.append(record["user_id"])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows)
    user_ids = sorted(set(y_rows))
    return X, y, user_ids


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class ZScoreScaler:
    """Minimal StandardScaler replacement (fit on train, apply to any set)."""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1e-8  # avoid divide-by-zero on constant features
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# 1. Identification: kNN + leave-one-out cross-validation
# ---------------------------------------------------------------------------

def sample_distances(train_X, query):
    """Manhattan distance from one query to every training row.

    Manhattan rather than Euclidean: squaring lets a single wildly-off
    feature dominate the sum, and with 26 features estimated from a few
    dozen keystrokes there is usually at least one such feature per sample.
    Summing absolute deviations lets the other 25 outvote it, which measured
    consistently better than Euclidean on the cross-phrase benchmark.
    """
    return np.abs(train_X - query).sum(axis=1)


def knn_predict(train_X, train_y, query, k=3):
    """Label of one query vector by distance-weighted k-nearest-neighbour.

    Each neighbour votes with weight 1/distance, so a near-identical sample
    counts for more than one that merely scraped into the top k. Plain
    majority also ties whenever k splits evenly across users and resolves
    the tie by label order, which is arbitrary; weighting removes that.
    """
    dists = sample_distances(train_X, query)
    nearest_idx = np.argsort(dists)[:k]

    votes = {}
    for i in nearest_idx:
        votes[train_y[i]] = votes.get(train_y[i], 0.0) + 1.0 / (dists[i] + 1e-9)
    return max(votes, key=votes.get)


def loocv_identification_accuracy(X, y, k=3):
    """
    Leave-one-out cross-validation: for each sample, train on every other
    sample and try to predict its user_id. Returns overall accuracy and a
    confusion matrix (as a dict of dicts: true_user -> predicted_user -> count).
    """
    n = len(X)
    correct = 0
    user_ids = sorted(set(y))
    confusion = {u: {v: 0 for v in user_ids} for u in user_ids}

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        scaler = ZScoreScaler().fit(X[train_mask])
        train_X = scaler.transform(X[train_mask])
        query = scaler.transform(X[i : i + 1])[0]

        pred = knn_predict(train_X, y[train_mask], query, k=k)
        confusion[y[i]][pred] += 1
        if pred == y[i]:
            correct += 1

    accuracy = correct / n
    return accuracy, confusion


def print_confusion_matrix(confusion: dict, user_ids: list):
    header = "true\\pred".ljust(12) + "".join(u[-2:].rjust(6) for u in user_ids)
    print(header)
    for u in user_ids:
        row = u.ljust(12) + "".join(str(confusion[u][v]).rjust(6) for v in user_ids)
        print(row)


# ---------------------------------------------------------------------------
# 2. Verification: per-user templates + genuine/impostor distance scores
# ---------------------------------------------------------------------------

#: A per-user template is not just a point but a point plus a spread. The
#: spread floor is a fraction of the population spread for that feature: a
#: user enrolled from two samples can show a near-zero spread on a feature
#: purely by luck, and dividing by that would make one lucky coincidence
#: outweigh every other feature.
SPREAD_FLOOR = 0.25


class Template:
    """One enrolled user, as the decision layer sees them.

    mean  where this user sits in normalized feature space
    std   how much they vary, per feature, across their own enrollments
    """

    __slots__ = ("user_id", "mean", "std", "n")

    def __init__(self, user_id, mean, std, n):
        self.user_id = user_id
        self.mean = mean
        self.std = std
        self.n = n

    def __repr__(self):
        return f"Template({self.user_id!r}, n={self.n})"


def template_distance(query, template: Template) -> float:
    """Scaled Manhattan distance from a sample to a user's template.

    Each feature's deviation is divided by how much *that user* naturally
    varies on it, then averaged. This is the part that plain Euclidean
    distance to a mean vector gets wrong: someone whose flight times are
    metronomic should be rejected for a deviation that would be unremarkable
    in someone erratic, and the population-level z-score cannot express that
    because it only knows the spread across everyone.

    Dividing by the per-user spread measured EER on the cross-phrase
    benchmark from 7.98% down to 5.45%; it is the single biggest win in the
    decision layer.
    """
    return float(np.mean(np.abs(query - template.mean) / template.std))


def build_templates(X, y, scaler: ZScoreScaler):
    """One Template per user, from their normalized enrollment samples.

    Returns dict: user_id -> Template.
    """
    Xn = scaler.transform(X)
    population_std = Xn.std(axis=0)
    templates = {}
    for u in sorted(set(y)):
        rows = Xn[y == u]
        spread = rows.std(axis=0) if len(rows) > 1 else np.zeros(Xn.shape[1])
        spread = np.maximum(spread, SPREAD_FLOOR * population_std + 1e-6)
        templates[u] = Template(u, rows.mean(axis=0), spread, len(rows))
    return templates


def verification_scores(X, y, user_ids):
    """
    Leave-one-out style: for each sample, build that user's template from
    their *other* samples only (no leakage), then compute:
      - genuine_scores: distance(sample, own template)
      - impostor_scores: distance(sample, every other user's template)

    Lower distance = more similar = more likely genuine.
    """
    genuine_scores = []
    impostor_scores = []

    for i in range(len(X)):
        this_user = y[i]

        # fit scaler on everyone else's data to avoid leaking this sample's
        # own statistics into normalization
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[i] = False
        scaler = ZScoreScaler().fit(X[train_mask])

        query = scaler.transform(X[i : i + 1])[0]
        templates = build_templates(X[train_mask], y[train_mask], scaler)

        for u in user_ids:
            if u not in templates:
                continue
            dist = template_distance(query, templates[u])
            if u == this_user:
                genuine_scores.append(dist)
            else:
                impostor_scores.append(dist)

    return np.array(genuine_scores), np.array(impostor_scores)


def equal_error_rate(genuine_scores, impostor_scores, n_thresholds=500):
    """
    Sweep a distance threshold and find where False Accept Rate (impostor
    scores below threshold, i.e. wrongly accepted) equals False Reject Rate
    (genuine scores above threshold, i.e. wrongly rejected). Returns
    (eer, threshold_at_eer).
    """
    lo = min(genuine_scores.min(), impostor_scores.min())
    hi = max(genuine_scores.max(), impostor_scores.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    best_eer = None
    best_thresh = None
    for t in thresholds:
        far = np.mean(impostor_scores <= t)   # impostors accepted (dist small)
        frr = np.mean(genuine_scores > t)      # genuine users rejected
        if best_eer is None or abs(far - frr) < abs(best_eer[0] - best_eer[1]):
            best_eer = (far, frr)
            best_thresh = t

    eer = (best_eer[0] + best_eer[1]) / 2.0
    return eer, best_thresh


if __name__ == "__main__":
    X, y, user_ids = build_dataset()
    print(f"Loaded {len(X)} samples across {len(user_ids)} users: {user_ids}\n")

    print("=== Identification (LOOCV kNN, k=3) ===")
    acc, confusion = loocv_identification_accuracy(X, y, k=3)
    print(f"Accuracy: {acc:.3f} ({int(acc * len(X))}/{len(X)})\n")
    print_confusion_matrix(confusion, user_ids)

    print("\n=== Verification (per-user templates, leave-one-out) ===")
    genuine, impostor = verification_scores(X, y, user_ids)
    print(f"Genuine scores:  mean={genuine.mean():.3f}  std={genuine.std():.3f}  n={len(genuine)}")
    print(f"Impostor scores: mean={impostor.mean():.3f}  std={impostor.std():.3f}  n={len(impostor)}")

    eer, thresh = equal_error_rate(genuine, impostor)
    print(f"\nEqual Error Rate (EER): {eer * 100:.2f}%  (threshold ~= {thresh:.3f})")
