"""
synthetic_data_generator.py

Generates synthetic keystroke-dynamics data for N simulated users typing a
fixed phrase, so the rest of the pipeline (signal construction, FFT features,
denoising, matching, classification) can be built and tested before real
human data is collected.

Each simulated user has a distinct underlying "typing profile":
    - a mean dwell time + std dev (per-user, roughly constant across keys)
    - a mean flight time + std dev
    - a small per-key bias (some keys are just slower for that user)

Output format (JSON), matches what the browser capture tool will produce:
{
  "user_id": "user_00",
  "phrase": "the quick brown fox",
  "samples": [
      {
        "events": [
          {"key": "t", "type": "down", "t": 0.0},
          {"key": "t", "type": "up",   "t": 0.083},
          {"key": "h", "type": "down", "t": 0.145},
          ...
        ]
      },
      ... (one dict per repetition)
  ]
}

Run this file directly to generate a full dataset under ./data/synthetic/.
"""

import json
import os
import random

import numpy as np

from keyloggd.paths import SYNTHETIC_DIR, SYNTHETIC_MULTI_DIR

PHRASE = "the quick brown fox"  # fixed enrollment/login phrase
N_USERS = 6
N_SAMPLES_PER_USER = 15
OUT_DIR = SYNTHETIC_DIR


def _chars_no_space(phrase: str) -> str:
    return phrase  # keep spaces as real keys; they have their own dwell/flight too


def make_user_profile(rng: np.random.Generator, phrase: str) -> dict:
    """
    Create a random but self-consistent typing profile for one synthetic user.
    """
    base_dwell = rng.uniform(0.07, 0.16)      # seconds, average key hold time
    base_flight = rng.uniform(0.08, 0.30)     # seconds, average gap between keys
    dwell_std = rng.uniform(0.01, 0.03)
    flight_std = rng.uniform(0.02, 0.06)

    # per-character multiplicative bias, so some keys are consistently
    # slower/faster for this user (e.g. pinky-finger keys)
    unique_chars = sorted(set(phrase))
    char_bias = {c: rng.uniform(0.8, 1.3) for c in unique_chars}

    return {
        "base_dwell": base_dwell,
        "base_flight": base_flight,
        "dwell_std": dwell_std,
        "flight_std": flight_std,
        "char_bias": char_bias,
    }


def generate_sample(rng: np.random.Generator, phrase: str, profile: dict) -> dict:
    """
    Generate one realization (one typed repetition) of the phrase for a given
    user profile, as a list of keydown/keyup events with timestamps.
    """
    events = []
    t = 0.0

    for i, ch in enumerate(phrase):
        bias = profile["char_bias"].get(ch, 1.0)

        # dwell time for this specific keystroke
        dwell = max(0.02, rng.normal(profile["base_dwell"] * bias, profile["dwell_std"]))

        key_down = t
        key_up = key_down + dwell
        events.append({"key": ch, "type": "down", "t": round(key_down, 5)})
        events.append({"key": ch, "type": "up", "t": round(key_up, 5)})

        # flight time to the next key (skip after last char)
        if i < len(phrase) - 1:
            flight = max(0.01, rng.normal(profile["base_flight"] * bias, profile["flight_std"]))
            t = key_up + flight
        else:
            t = key_up

    return {"events": events}


def generate_dataset(
    n_users: int = N_USERS,
    n_samples: int = N_SAMPLES_PER_USER,
    phrase: str = PHRASE,
    seed: int = 42,
    out_dir: str = OUT_DIR,
):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    for u in range(n_users):
        user_id = f"user_{u:02d}"
        profile = make_user_profile(rng, phrase)
        samples = [generate_sample(rng, phrase, profile) for _ in range(n_samples)]

        record = {"user_id": user_id, "phrase": phrase, "samples": samples}
        out_path = os.path.join(out_dir, f"{user_id}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"wrote {out_path}  ({n_samples} samples)")


MULTI_OUT_DIR = SYNTHETIC_MULTI_DIR


def generate_multiphrase_dataset(
    phrases: list,
    n_users: int = N_USERS,
    n_samples_per_phrase: int = 6,
    seed: int = 7,
    out_dir: str = MULTI_OUT_DIR,
):
    """Every simulated user types *every* phrase, for free-text evaluation.

    The single-phrase generator above gives each user a per-character bias built
    from that one phrase, which is fine when the phrase never changes. Here the
    profile is built once over the union of all the phrases' characters instead,
    so the same person carries the same per-key habits from sentence to sentence
    -- which is exactly the thing a cross-sentence test has to measure. (The
    per-sample generator already falls back to a neutral 1.0 bias for a
    character it has not seen, so a shared alphabet only makes the profile more
    complete, never less valid.)

    Written to its own directory: data/synthetic/ holds real captures and the
    app's live enrolled set, and must not be overwritten by a test fixture.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    alphabet = "".join(sorted(set("".join(phrases))))

    for u in range(n_users):
        user_id = f"user_{u:02d}"
        profile = make_user_profile(rng, alphabet)

        samples = []
        for phrase in phrases:
            for _ in range(n_samples_per_phrase):
                sample = generate_sample(rng, phrase, profile)
                # per-sample phrase, the schema the capture tool writes
                sample["phrase"] = phrase
                samples.append(sample)

        record = {"user_id": user_id, "phrase": phrases[0], "samples": samples}
        out_path = os.path.join(out_dir, f"{user_id}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"wrote {out_path}  ({len(samples)} samples over "
              f"{len(phrases)} phrases)")


if __name__ == "__main__":
    generate_dataset()
    print(f"\nDone. Synthetic dataset for {N_USERS} users written to: {OUT_DIR}")
