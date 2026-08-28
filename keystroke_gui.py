"""
Keystroke Classifier - Minimal Tkinter GUI
============================================

A simple desktop UI wrapped around `train_keystroke_classifier.py`. Lets you:
  1. Pick a dataset folder and train a model (with live log output).
  2. Load a trained model.
  3. Record a short clip from your mic and get a live prediction, or
     classify an existing .wav file.

REQUIREMENTS
------------
    pip install librosa scikit-learn numpy soundfile joblib sounddevice --break-system-packages

This file must sit in the same folder as `train_keystroke_classifier.py`,
since it imports the feature extraction / train / predict logic from it
rather than duplicating it.

RUN
---
    python keystroke_gui.py
"""

import os
import sys
import threading
import queue
import tempfile

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Local module (must be alongside this file)
import train_keystroke_classifier as kc


class KeystrokeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keystroke Sound Classifier")
        self.geometry("560x520")
        self.minsize(520, 480)

        self.data_dir = tk.StringVar()
        self.model_path = tk.StringVar(value=os.path.abspath("keystroke_model.joblib"))
        self.record_seconds = tk.DoubleVar(value=0.6)
        self.status = tk.StringVar(value="Ready.")

        self.log_queue = queue.Queue()
        self.bundle = None  # loaded model bundle, lazily populated

        self._build_layout()
        self.after(150, self._poll_log_queue)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        pad = {"padx": 10, "pady": 6}

        # --- Training section ---
        train_frame = ttk.LabelFrame(self, text="1. Train a model")
        train_frame.pack(fill="x", **pad)

        row = ttk.Frame(train_frame)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Dataset folder:").pack(side="left")
        ttk.Entry(row, textvariable=self.data_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse...", command=self._choose_data_dir).pack(side="left")

        row2 = ttk.Frame(train_frame)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="Save model as:").pack(side="left")
        ttk.Entry(row2, textvariable=self.model_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="Browse...", command=self._choose_model_save_path).pack(side="left")

        ttk.Button(train_frame, text="Train Model", command=self._start_training).pack(
            padx=8, pady=(4, 8), anchor="w"
        )

        # --- Model loading section ---
        model_frame = ttk.LabelFrame(self, text="2. Load a trained model")
        model_frame.pack(fill="x", **pad)

        row3 = ttk.Frame(model_frame)
        row3.pack(fill="x", padx=8, pady=6)
        ttk.Button(row3, text="Load Model...", command=self._load_model).pack(side="left")
        self.model_status_label = ttk.Label(row3, text="No model loaded.")
        self.model_status_label.pack(side="left", padx=10)

        # --- Prediction section ---
        predict_frame = ttk.LabelFrame(self, text="3. Predict a keystroke")
        predict_frame.pack(fill="x", **pad)

        row4 = ttk.Frame(predict_frame)
        row4.pack(fill="x", padx=8, pady=4)
        ttk.Label(row4, text="Record length (s):").pack(side="left")
        ttk.Spinbox(
            row4, from_=0.2, to=3.0, increment=0.1, textvariable=self.record_seconds, width=6
        ).pack(side="left", padx=6)
        ttk.Button(row4, text="Record & Predict", command=self._start_record_predict).pack(
            side="left", padx=10
        )
        ttk.Button(row4, text="Predict from File...", command=self._predict_from_file).pack(side="left")

        self.prediction_label = ttk.Label(
            predict_frame, text="Prediction: -", font=("TkDefaultFont", 14, "bold")
        )
        self.prediction_label.pack(padx=8, pady=(6, 0), anchor="w")

        self.candidates_label = ttk.Label(predict_frame, text="", justify="left")
        self.candidates_label.pack(padx=8, pady=(0, 8), anchor="w")

        # --- Log / status ---
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)

        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill="x", side="bottom"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.configure(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(150, self._poll_log_queue)

    def _set_status(self, text):
        self.status.set(text)

    class _LogRedirector:
        """Redirects print() calls into the GUI log while a task runs."""
        def __init__(self, log_fn):
            self.log_fn = log_fn

        def write(self, text):
            text = text.rstrip("\n")
            if text:
                self.log_fn(text)

        def flush(self):
            pass

    # ------------------------------------------------------------------
    # Actions: choosing paths
    # ------------------------------------------------------------------
    def _choose_data_dir(self):
        path = filedialog.askdirectory(title="Select dataset folder")
        if path:
            self.data_dir.set(path)

    def _choose_model_save_path(self):
        path = filedialog.asksaveasfilename(
            title="Save model as",
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib")],
        )
        if path:
            self.model_path.set(path)

    # ------------------------------------------------------------------
    # Actions: training
    # ------------------------------------------------------------------
    def _start_training(self):
        data_dir = self.data_dir.get().strip()
        model_out = self.model_path.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("Error", "Please choose a valid dataset folder.")
            return
        if not model_out:
            messagebox.showerror("Error", "Please choose where to save the model.")
            return

        self._set_status("Training... this may take a while.")
        thread = threading.Thread(target=self._run_training, args=(data_dir, model_out), daemon=True)
        thread.start()

    def _run_training(self, data_dir, model_out):
        old_stdout = sys.stdout
        sys.stdout = self._LogRedirector(self._log)
        try:
            kc.train(data_dir, model_out)
            self._log(f"\nDone. Model saved to: {model_out}")
            self.model_path.set(model_out)
            self._auto_load_model(model_out)
            self._set_status("Training complete.")
        except Exception as e:
            self._log(f"[ERROR] Training failed: {e}")
            self._set_status("Training failed. See log.")
        finally:
            sys.stdout = old_stdout

    # ------------------------------------------------------------------
    # Actions: loading a model
    # ------------------------------------------------------------------
    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if path:
            self._auto_load_model(path)

    def _auto_load_model(self, path):
        try:
            import joblib
            self.bundle = joblib.load(path)
            n_classes = len(self.bundle["label_encoder"].classes_)
            self.model_status_label.config(
                text=f"Loaded: {os.path.basename(path)} ({n_classes} keys)"
            )
            self._log(f"Loaded model: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")

    # ------------------------------------------------------------------
    # Actions: prediction
    # ------------------------------------------------------------------
    def _require_model(self):
        if self.bundle is None:
            messagebox.showwarning("No model", "Load or train a model first.")
            return False
        return True

    def _predict_from_file(self):
        if not self._require_model():
            return
        path = filedialog.askopenfilename(
            title="Select audio clip",
            filetypes=[("Audio", "*.wav *.mp3"), ("All files", "*.*")],
        )
        if not path:
            return
        self._set_status("Predicting...")
        threading.Thread(target=self._run_predict_on_file, args=(path,), daemon=True).start()

    def _run_predict_on_file(self, path):
        old_stdout = sys.stdout
        sys.stdout = self._LogRedirector(self._log)
        try:
            label, top = self._predict_with_bundle(path)
            self._display_prediction(label, top)
            self._set_status("Prediction complete.")
        except Exception as e:
            self._log(f"[ERROR] Prediction failed: {e}")
            self._set_status("Prediction failed. See log.")
        finally:
            sys.stdout = old_stdout

    def _start_record_predict(self):
        if not self._require_model():
            return
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "Recording requires the 'sounddevice' package:\n"
                "pip install sounddevice --break-system-packages",
            )
            return

        self._set_status("Recording...")
        threading.Thread(target=self._run_record_predict, daemon=True).start()

    def _run_record_predict(self):
        import sounddevice as sd
        import soundfile as sf

        sr = self.bundle["sample_rate"] if self.bundle else 16000
        duration = self.record_seconds.get()
        try:
            self._log(f"Recording {duration:.1f}s at {sr} Hz...")
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)
            self._log(f"Recorded to temp file: {tmp_path}")

            label, top = self._predict_with_bundle(tmp_path)
            self._display_prediction(label, top)
            self._set_status("Prediction complete.")
        except Exception as e:
            self._log(f"[ERROR] Recording/prediction failed: {e}")
            self._set_status("Failed. See log.")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _predict_with_bundle(self, clip_path):
        """Runs feature extraction + prediction using the currently loaded bundle."""
        clf = self.bundle["model"]
        scaler = self.bundle["scaler"]
        encoder = self.bundle["label_encoder"]
        sr = self.bundle["sample_rate"]

        audio, sr_ = kc.load_audio(clip_path, sr=sr)
        audio = kc.extract_keystroke_segment(audio, sr_)
        feats = kc.extract_features(audio, sr_).reshape(1, -1)
        feats_scaled = scaler.transform(feats)

        pred = clf.predict(feats_scaled)[0]
        proba = clf.predict_proba(feats_scaled)[0]
        label = encoder.inverse_transform([pred])[0]

        import numpy as np
        top_idx = np.argsort(proba)[::-1][:5]
        top = [(encoder.classes_[i], proba[i]) for i in top_idx]
        return label, top

    def _display_prediction(self, label, top):
        self.prediction_label.config(text=f"Prediction: {label}")
        lines = [f"{name:>8s}  {p*100:5.1f}%" for name, p in top]
        self.candidates_label.config(text="\n".join(lines))
if __name__ == "__main__":
    app = KeystrokeGUI()
    app.mainloop()