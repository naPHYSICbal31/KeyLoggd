"""
signal_construction.py

Converts raw (key, event_type, timestamp) event lists -- exactly what both the
synthetic generator and the browser capture tool produce -- into two discrete
1-D signals per typed sample:

    dwell[n]  = duration key n was held down       (keyup - keydown)
    flight[n] = gap between releasing key n and
                pressing key n+1                    (keydown[n+1] - keyup[n])

These are the raw discrete-time signals that everything downstream (FFT
features, denoising, convolution matching) operates on.

Conceptual note (for the report): the *true* underlying process is a
continuous-time impulse train x(t) = sum_i delta(t - t_i) where t_i are the
keydown times. What we build here -- dwell[n], flight[n] -- is a sampled,
event-indexed representation of that process (sampled at the *events*
themselves rather than at fixed time steps), which is what we then treat as a
discrete signal for DFT/FFT purposes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KeystrokeSignal:
    dwell: np.ndarray          # dwell time per keystroke, seconds
    flight: np.ndarray         # flight time between consecutive keystrokes, seconds
    keys: list                 # the key sequence, same order as dwell
    raw_events: list           # original event dicts, kept for debugging/plots


def events_to_signal(events: list) -> KeystrokeSignal:
    """
    events: list of dicts like {"key": "t", "type": "down"/"up", "t": <float seconds>}
            must be in chronological order, one down+up pair per keystroke.
    """
    downs = {}
    dwell_list = []
    flight_list = []
    keys_list = []

    last_up_time = None

    # Walk events assuming standard down->up ordering per key (true for our
    # synthetic generator and for the capture tool below).
    i = 0
    while i < len(events):
        ev = events[i]
        if ev["type"] == "down":
            key = ev["key"]
            down_t = ev["t"]

            # find the matching "up" for this same key, scanning forward
            up_t = None
            for j in range(i + 1, len(events)):
                if events[j]["key"] == key and events[j]["type"] == "up":
                    up_t = events[j]["t"]
                    break
            if up_t is None:
                i += 1
                continue  # malformed / incomplete event, skip

            dwell = up_t - down_t
            dwell_list.append(dwell)
            keys_list.append(key)

            if last_up_time is not None:
                flight_list.append(down_t - last_up_time)

            last_up_time = up_t
        i += 1

    return KeystrokeSignal(
        dwell=np.array(dwell_list, dtype=float),
        flight=np.array(flight_list, dtype=float),
        keys=keys_list,
        raw_events=events,
    )


def sample_to_signal(sample: dict) -> KeystrokeSignal:
    """Convenience wrapper: sample is one entry from the JSON 'samples' list."""
    return events_to_signal(sample["events"])


def load_user_signals(user_json_record: dict) -> list:
    """
    user_json_record: the loaded JSON dict for one user, e.g.
        {"user_id": ..., "phrase": ..., "samples": [...]}
    Returns a list of KeystrokeSignal, one per repetition/sample.
    """
    return [sample_to_signal(s) for s in user_json_record["samples"]]


if __name__ == "__main__":
    # quick smoke test using the synthetic generator
    from tools.synthetic_data_generator import generate_sample, make_user_profile

    rng = np.random.default_rng(0)
    profile = make_user_profile(rng, "the quick brown fox")
    sample = generate_sample(rng, "the quick brown fox", profile)

    sig = sample_to_signal(sample)
    print("keys:  ", sig.keys)
    print("dwell: ", np.round(sig.dwell, 4))
    print("flight:", np.round(sig.flight, 4))
