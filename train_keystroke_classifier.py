"""
Keystroke Acoustic Classifier
==============================

Trains a model to classify which key was pressed based on the sound of the
keystroke (a "acoustic side-channel" / audio classification task).

EXPECTED DATA LAYOUT
---------------------
Point this script at a directory structured like:

    dataset/
        a/
            a_001.wav
            a_002.wav
            ...
        b/
            b_001.wav
            ...
        space/
            space_001.wav
            ...
        enter/
            ...

Each sub-folder name is the label (the key), and it contains short .wav
clips of that key being pressed (ideally each clip trimmed to just the
keystroke, e.g. 200-500ms). You need to record and label this data yourself
- this script does not come with a dataset.

HOW TO COLLECT DATA
--------------------
- Record with a decent microphone close to the keyboard, consistent
  position, consistent room noise.
- For each key press, save a short isolated clip (a simple onset-detection
  / silence-trimming step is included below via `extract_keystroke_segment`
  if you record continuous audio instead of discrete clips).
- Aim for at least 30-50 examples per key for a usable baseline, more is
  better. Keep recording conditions (mic, distance, keyboard, surface)
  consistent between train data and any data you evaluate on later -
  models trained this way generalize poorly across different keyboards,
  mics, or environments.

PIPELINE
--------
1. Load audio clips per key.
2. Extract features (MFCCs + spectral descriptors).
3. Train/test split.
4. Train a classifier (RandomForest baseline; optional small CNN on
   spectrograms if PyTorch is available).
5. Evaluate + save the trained model.

USAGE
-----
    pip install librosa scikit-learn numpy soundfile joblib --break-system-packages
    python train_keystroke_classifier.py --data_dir ./dataset --model_out keystroke_model.joblib

    # Then predict on a new clip:
    python train_keystroke_classifier.py --predict path/to/clip.wav --model_out keystroke_model.joblib
"""

import argparse
import os
import sys
import glob
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Audio utilities
# --------------------------------------------------------------------------

def load_audio(path, sr=16000):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_keystroke_segment(y, sr, pre_ms=20, post_ms=180, top_db=30):
    """
    If a clip contains silence around the keystroke (or you fed in a longer
    continuous recording), isolate the loudest onset and crop a fixed
    window around it. This makes clip length/alignment consistent across
    examples, which matters a lot for feature quality.
    """
    import librosa
    # Find non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        onset_sample = int(np.argmax(np.abs(y)))
    else:
        # Take the loudest interval's peak as the onset
        peak_val = -1
        onset_sample = 0
        for start, end in intervals:
            seg = y[start:end]
            local_peak = np.max(np.abs(seg)) if len(seg) else 0
            if local_peak > peak_val:
                peak_val = local_peak
                onset_sample = start + int(np.argmax(np.abs(seg)))

    pre = int(sr * pre_ms / 1000)
    post = int(sr * post_ms / 1000)
    start = max(0, onset_sample - pre)
    end = min(len(y), onset_sample + post)
    segment = y[start:end]

    # Pad to fixed length so all feature vectors line up
    target_len = pre + post
    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)))
    else:
        segment = segment[:target_len]
    return segment


def extract_features(y, sr, n_mfcc=20):
    """
    Extract a fixed-length feature vector combining MFCCs (timbre),
    spectral centroid/bandwidth/rolloff (brightness/shape), zero-crossing
    rate (percussiveness), and RMS energy envelope stats — all useful for
    telling apart the short, percussive clicks of different keys.
    """
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    def stats(x):
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])

    feature_vec = np.concatenate([
        stats(mfcc),
        stats(mfcc_delta),
        stats(spec_centroid),
        stats(spec_bandwidth),
        stats(spec_rolloff),
        stats(zcr),
        stats(rms),
    ])
    return feature_vec


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def build_dataset(data_dir, sr=16000, auto_segment=True):
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if not labels:
        raise ValueError(f"No label sub-folders found in {data_dir}")

    X, y_labels = [], []
    for label in labels:
        files = glob.glob(os.path.join(data_dir, label, "*.wav"))
        files += glob.glob(os.path.join(data_dir, label, "*.mp3"))
        if not files:
            print(f"  [warn] no audio files found for label '{label}'")
            continue
        print(f"  loading '{label}': {len(files)} files")
        for f in files:
            try:
                audio, sr_ = load_audio(f, sr=sr)
                if auto_segment:
                    audio = extract_keystroke_segment(audio, sr_)
                feats = extract_features(audio, sr_)
                X.append(feats)
                y_labels.append(label)
            except Exception as e:
                print(f"    [skip] {f}: {e}")

    if not X:
        raise ValueError("No usable audio was loaded. Check your dataset directory.")

    return np.array(X), np.array(y_labels)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(data_dir, model_out, sr=16000):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    print(f"Building dataset from {data_dir} ...")
    X, y_labels = build_dataset(data_dir, sr=sr)
    print(f"Total examples: {len(X)}, classes: {sorted(set(y_labels))}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    bundle = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": encoder,
        "sample_rate": sr,
    }
    joblib.dump(bundle, model_out)
    print(f"\nSaved model to {model_out}")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict(clip_path, model_path):
    import joblib

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    scaler = bundle["scaler"]
    encoder = bundle["label_encoder"]
    sr = bundle["sample_rate"]

    audio, sr_ = load_audio(clip_path, sr=sr)
    audio = extract_keystroke_segment(audio, sr_)
    feats = extract_features(audio, sr_).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    pred = clf.predict(feats_scaled)[0]
    proba = clf.predict_proba(feats_scaled)[0]
    label = encoder.inverse_transform([pred])[0]

    top5_idx = np.argsort(proba)[::-1][:5]
    print(f"Predicted key: {label}")
    print("Top candidates:")
    for i in top5_idx:
        print(f"  {encoder.classes_[i]:>8s}  {proba[i]*100:5.1f}%")

    return label


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train/use a keystroke sound classifier")
    parser.add_argument("--data_dir", type=str, help="Path to dataset dir with one sub-folder per key")
    parser.add_argument("--model_out", type=str, default="keystroke_model.joblib", help="Path to save/load model")
    parser.add_argument("--predict", type=str, help="Path to a .wav clip to classify with a trained model")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate for loading audio")
    args = parser.parse_args()

    if args.predict:
        if not os.path.exists(args.model_out):
            sys.exit(f"Model file not found: {args.model_out}. Train first.")
        predict(args.predict, args.model_out)
    elif args.data_dir:
        train(args.data_dir, args.model_out, sr=args.sr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()