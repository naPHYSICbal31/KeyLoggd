"""
timing_features.py

The feature vector the identification decision is actually built on: robust
statistics of the dwell/flight streams, plus the handful of per-key habits
that survive a change of sentence.

Why this exists alongside fft_features (for the report):

    fft_features summarises each spectrum into five numbers -- centroid,
    three band-energy ratios, total energy -- and that vector was what the
    classifier used. Measured against a free-text benchmark it is the weak
    part of the pipeline, for three reasons that all point the same way:

      * A sample is 20-40 keystrokes. The real half-spectrum is then ~17
        bins estimated from ~30 noisy points, so the band ratios are mostly
        estimation noise, and collapsing them to three numbers keeps the
        noise while discarding whatever structure was there.
      * Zero-padding to a fixed length makes the spectrum a function of how
        long the typed text was as much as of how the person typed. On a
        fixed enrollment phrase that is invisible; on free text it moves
        every sample for a reason that has nothing to do with the typist.
      * total_energy is dominated by a single think-pause: a 1.3s gap is
        ~20x a typical 0.07s one, so one hesitation outweighs the entire
        rest of the sample.

    Swapping those ten spectral numbers for the statistics here took
    leave-one-phrase-out identification from 64.6% to 92.5% and verification
    EER from 25.0% to 5.8% on the cross-phrase benchmark. The spectral
    features were tried again on top of these -- raw, compressed, and with
    2/4/6 bins kept -- and cost accuracy every time, so they are not in the
    vector. fft_features stays in the pipeline: denoise uses it, and the
    explain view still shows the spectrum, which is the clearest picture of
    what typing rhythm *is* even where it is not the sharpest feature.

Design rules the blocks below follow:

    Robust before sharp.   Medians, IQRs and median-absolute deviations
                           rather than means and standard deviations: one
                           pause should not move the whole vector.
    Sign-aware.            Flight times go negative (see _alog). Nothing
                           here may assume a positive timing.
    Speed-invariant where  The same person types faster when fresh and
    it is free.            slower when tired, so ratios and shape features
                           are preferred and absolute speed is carried by
                           its own few dimensions rather than smeared
                           through every one.
"""

import numpy as np

#: Scale for the asinh compression, in seconds -- about one typical flight
#: time, so ordinary typing lands in the transform's near-linear region and
#: only pauses get compressed.
ALOG_SCALE = 0.05


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _alog(x, scale: float = ALOG_SCALE):
    """Sign-aware log: asinh(x / scale).

    Flight times go *negative* on a real keyboard. A fast typist presses the
    next key before releasing the last one -- rollover -- and in the captures
    here that is 18% of one user's flights and 7% of another's. The rate
    itself is a per-person habit, so it is signal, not noise to be clipped.

    log() cannot represent it: clamping at a positive floor maps every
    rollover onto the same large negative number, which destroys the mean and
    the spread of the one signal that best separates fast typists from slow
    ones. asinh is linear near zero and logarithmic in the tails, so it
    compresses think-pauses the way a log would while passing rollover
    through with its sign intact.
    """
    return np.arcsinh(np.asarray(x, dtype=float) / scale)


def _is_space(key) -> bool:
    """True for the spacebar however the capture layer spelled it."""
    if not isinstance(key, str):
        return False
    return key == " " or key.strip().lower() in ("space", "spacebar")


# ---------------------------------------------------------------------------
# Block 1: the distribution of one timing stream
# ---------------------------------------------------------------------------

STATS_NAMES = [
    ("compressed mean", "aMu"),
    ("compressed spread", "aSd"),
    ("median", "med"),
    ("interquartile range", "iqr"),
    ("median abs deviation", "mad"),
    ("10-90 percentile span", "p90"),
    ("robust variability", "rcv"),
    ("long-outlier fraction", "outl"),
]

#: Extra name used only for a stream that can legitimately go negative.
SIGNED_NAME = ("rollover fraction", "roll")


def stats_block(x, signed: bool = False):
    """Robust descriptors of one timing stream (dwell or flight).

    signed=True adds the negative fraction, which is meaningful for flight
    (key rollover) and always zero for dwell -- a key cannot be released
    before it is pressed -- so dwell omits it rather than carrying a constant
    dimension that only adds noise to the normalisation.
    """
    width = len(STATS_NAMES) + (1 if signed else 0)
    x = _finite(x)
    if len(x) == 0:
        return [0.0] * width

    lx = _alog(x)
    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
    mad = float(np.median(np.abs(x - q50)))

    out = [
        float(lx.mean()),                        # speed, in a symmetric domain
        float(lx.std()),                         # scale-free variability
        float(q50),
        float(q75 - q25),                        # spread immune to one pause
        mad,
        float(q90 - q10),
        float(mad / (abs(q50) + 1e-9)),          # robust coefficient of variation
        float(np.mean(x > q50 + 2.0 * mad)),     # how often they hesitate
    ]
    if signed:
        out.append(float(np.mean(x < 0.0)))
    return out


# ---------------------------------------------------------------------------
# Block 2: habits tied to which key, not just to the timing stream
# ---------------------------------------------------------------------------

KEYCLASS_NAMES = [
    ("space hold", "spDw"),
    ("letter hold", "ltDw"),
    ("space vs letter hold", "s:l"),
    ("flight after space", "aSp"),
    ("flight within word", "inW"),
    ("word-start vs in-word gap", "a:i"),
]


def keyclass_block(dwell, flight, keys):
    """Per-key habits that any English sentence can supply.

    A full per-character table would be the stronger feature and is what the
    literature uses on fixed passwords, but a 30-key free-text sample touches
    each letter once or twice, so nearly every cell would be empty or
    estimated from a single observation.

    Space is the exception: it is the most frequent key in any English text
    by a wide margin (54 of one user's 276 keystrokes here), so space-vs-
    letter hold time and the flight that starts a new word are measurable on
    a single sentence and carry real identity -- whether someone punches or
    brushes the spacebar, and whether they pause between words or run them
    together.
    """
    dwell = np.asarray(dwell, dtype=float)
    flight = np.asarray(flight, dtype=float)
    keys = list(keys)

    n = min(len(dwell), len(keys))
    if n == 0:
        return [0.0] * len(KEYCLASS_NAMES)
    dwell, keys = dwell[:n], keys[:n]

    space = np.array([_is_space(k) for k in keys])
    sp_d = _finite(dwell[space])
    lt_d = _finite(dwell[~space])
    sp_hold = float(np.median(sp_d)) if len(sp_d) else 0.0
    lt_hold = float(np.median(lt_d)) if len(lt_d) else 0.0

    # flight[i] is the gap between keystroke i and keystroke i+1, so the
    # class of the key it follows is space[i].
    m = min(len(flight), n - 1)
    after_space = _finite(flight[:m][space[:m]]) if m > 0 else np.array([])
    within = _finite(flight[:m][~space[:m]]) if m > 0 else np.array([])
    a = float(np.median(after_space)) if len(after_space) else 0.0
    w = float(np.median(within)) if len(within) else 0.0

    # Ratios are taken as differences of compressed values -- _alog(x) -
    # _alog(y) is a log-ratio for ordinary values but stays finite when the
    # denominator approaches zero. A raw quotient does not: a fast typist's
    # median within-word flight can sit at 0.017s, and dividing by that
    # turns a normal sample into an extreme outlier. See ratio_block.
    return [
        sp_hold,
        lt_hold,
        float(_alog(sp_hold) - _alog(lt_hold)),
        a,
        w,
        float(_alog(a) - _alog(w)),
    ]


# ---------------------------------------------------------------------------
# Block 3: cross-stream ratios, which survive a change of overall speed
# ---------------------------------------------------------------------------

RATIO_NAMES = [
    ("hold vs gap", "d:f"),
    ("duty cycle", "duty"),
    ("keys per second", "kps"),
]


def ratio_block(dwell, flight):
    """Relationships between the two streams.

    The same person types faster when fresh and slower when tired, which
    moves every absolute timing at once. A relationship between two of them
    barely moves, so these are the dimensions that carry identity across
    sessions rather than carrying the session.

    Every one of them is bounded. The obvious forms are not: median flight
    legitimately approaches zero for a fast typist with heavy rollover -- one
    real sample here sits at 0.017s against a 0.07s norm -- and a plain
    dwell/flight quotient turns that sample into a 5x outlier that no
    template can match. Written as a compressed difference and a normalised
    fraction instead, the same sample stays where it belongs.
    """
    d, f = _finite(dwell), _finite(flight)
    if len(d) == 0 or len(f) == 0:
        return [0.0] * len(RATIO_NAMES)

    dm = float(np.median(d))
    fm = float(np.median(f))

    # hold vs gap, as a difference of compressed values: a log-ratio for
    # ordinary timings, finite when the gap collapses to zero or goes
    # negative.
    hold_vs_gap = float(_alog(dm) - _alog(fm))

    # fraction of each key cycle spent holding, in [0, 1] by construction
    duty = float(dm / (dm + abs(fm) + 1e-9))

    # raw speed from the elapsed run, not from a reciprocal median: the sum
    # of both streams is the time the sample actually took, so this stays
    # positive and finite however the medians fall.
    elapsed = float(d.sum() + np.abs(f).sum())
    kps = float(len(d) / elapsed) if elapsed > 1e-9 else 0.0

    return [hold_vs_gap, duty, kps]


# ---------------------------------------------------------------------------
# The vector
# ---------------------------------------------------------------------------

#: (full name for the values panel, compact axis code) for every component,
#: in vector order. The UI reads this instead of keeping its own copy, so a
#: change here cannot leave the explain view mislabelling the numbers.
FEATURE_NAMES = (
    [(f"dwell {n}", f"d{s}") for n, s in STATS_NAMES]
    + [(f"flight {n}", f"f{s}") for n, s in STATS_NAMES]
    + [(f"flight {SIGNED_NAME[0]}", f"f{SIGNED_NAME[1]}")]
    + KEYCLASS_NAMES
    + RATIO_NAMES
)

N_FEATURES = len(FEATURE_NAMES)


def feature_vector(dwell, flight, keys=None) -> np.ndarray:
    """One vector per typed sample: the thing templates and distances live in.

    keys is the key sequence aligned with dwell (KeystrokeSignal.keys). It is
    optional so a caller holding only the two timing streams still works; the
    per-key block is then zero-filled and the vector keeps its width, which
    matters because the scaler and every template are indexed by position.
    """
    vec = (
        stats_block(dwell, signed=False)
        + stats_block(flight, signed=True)
        + keyclass_block(dwell, flight, keys if keys is not None else [])
        + ratio_block(dwell, flight)
    )
    out = np.asarray(vec, dtype=float)
    # A degenerate sample (one keystroke, a dropped event) can still produce
    # a nan or an inf through a ratio; the decision layer must never see one.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def signal_feature_vector(signal) -> np.ndarray:
    """feature_vector for a KeystrokeSignal, keys included."""
    return feature_vector(signal.dwell, signal.flight, signal.keys)


if __name__ == "__main__":
    from keyloggd.pipeline.signal_construction import sample_to_signal
    from tools.synthetic_data_generator import generate_sample, make_user_profile

    rng = np.random.default_rng(1)
    phrase = "the quick brown fox"
    sig = sample_to_signal(generate_sample(rng, phrase, make_user_profile(rng, phrase)))

    vec = signal_feature_vector(sig)
    print(f"{N_FEATURES} features, vector shape {vec.shape}")
    assert len(vec) == N_FEATURES
    for (long, short), v in zip(FEATURE_NAMES, vec):
        print(f"  {short:6s} {long:28s} {v:9.4f}")
