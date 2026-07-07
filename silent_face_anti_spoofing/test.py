# -*- coding: utf-8 -*-
"""
Combines the MiniFASNet anti-spoofing score (this folder) with a randomized
blink + smile challenge-response to decide REAL-and-live vs FAKE.

Blink/smile detection is ported from ../web/face_engine.py in the sibling
`facenet` repo (MediaPipe FaceLandmarker blendshapes: eyeBlinkLeft/Right,
mouthSmileLeft/Right) — reuses the already-downloaded
../models/face_landmarker.task rather than adding a new model.

Why combine them: a photo/video replay can still fake a blink or a smile on
camera, but the anti-spoof model keeps scoring its texture as fake across
most frames. A real person passes both. Either check failing fails the
whole session.

Usage:
    python challenge_test.py                    # interactive webcam
    python challenge_test.py --camera 1
    python challenge_test.py --max_seconds 15    # auto-quit (for scripted smoke tests)

Keys: 'q' quit, 'r' reset (reshuffle challenges, retry).
"""

import os
import random
import time
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from ai_edge_litert.interpreter import Interpreter

from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings("ignore")

MODEL_DIR_TFLITE = "./resources/anti_spoof_models_tflite"
LANDMARKER_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

BLINK_CLOSED_THRESHOLD = 0.5  # eyeBlinkLeft/Right blendshape score above this = closed
SMILE_THRESHOLD = 0.5  # mouthSmileLeft/Right blendshape score above this = smiling
FAKE_STREAK_LIMIT = 5  # consecutive fake anti-spoof reads -> abort as spoof
REAL_RATIO_PASS = 0.7  # fraction of frames scored "real" needed for a final REAL verdict


# ---- anti-spoof (this repo's MiniFASNet models) -------------------------


def load_antispoof_models(tflite_dir):
    loaded = []
    for model_name in os.listdir(tflite_dir):
        interpreter = Interpreter(model_path=os.path.join(tflite_dir, model_name))
        interpreter.allocate_tensors()
        loaded.append((model_name, interpreter))
    return loaded


def score_antispoof(loaded_models, image_cropper, frame, bbox):
    prediction = np.zeros((1, 3))
    for model_name, interpreter in loaded_models:
        h_input, w_input, _, scale = parse_model_name(model_name)
        img = image_cropper.crop(
            org_img=frame,
            bbox=bbox,
            scale=scale,
            out_w=w_input,
            out_h=h_input,
            crop=scale is not None,
        )
        # no /255 here -- this repo's ToTensor deliberately skips normalization
        # (src/data_io/functional.py), MiniFASNet is trained on raw [0, 255] float.
        input_tensor = img.transpose(2, 0, 1).astype(np.float32)[None, ...]
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"])
        exp = np.exp(logits - logits.max())
        prediction += exp / exp.sum()
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return label == 1, value  # is_real, confidence


# ---- face landmarks + blendshapes (ported from ../web/face_engine.py) --


class FaceMesh:
    def __init__(self):
        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
                output_face_blendshapes=True,
                num_faces=1,
            )
        )

    def analyze(self, bgr):
        h, w = bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._landmarker.detect(image)
        if not res.face_landmarks:
            return None
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bbox_xywh = (max(int(x1), 0), max(int(y1), 0), int(x2 - x1), int(y2 - y1))
        shapes = {s.category_name: s.score for s in res.face_blendshapes[0]}
        return bbox_xywh, shapes


# ---- randomized challenge state machine (mirrors the Flutter liveness
# detector's blink/smile challenges, order shuffled per session so a
# pre-scripted replay can't be made to match a fixed sequence) -----------


class ChallengeRunner:
    def __init__(self):
        self.challenges = ["blink", "smile"]
        random.shuffle(self.challenges)
        self.index = 0
        self._eyes_closed_seen = False

    @property
    def current(self):
        return self.challenges[self.index] if self.index < len(self.challenges) else None

    @property
    def done(self):
        return self.index >= len(self.challenges)

    def update(self, shapes):
        """Returns True if the current challenge just completed."""
        if self.done:
            return False
        challenge = self.challenges[self.index]
        completed = self._check_blink(shapes) if challenge == "blink" else self._check_smile(shapes)
        if completed:
            self.index += 1
            self._eyes_closed_seen = False
        return completed

    def _check_blink(self, shapes):
        left = shapes.get("eyeBlinkLeft", 0.0)
        right = shapes.get("eyeBlinkRight", 0.0)
        closed = left > BLINK_CLOSED_THRESHOLD and right > BLINK_CLOSED_THRESHOLD
        if closed:
            self._eyes_closed_seen = True
            return False
        return self._eyes_closed_seen

    def _check_smile(self, shapes):
        left = shapes.get("mouthSmileLeft", 0.0)
        right = shapes.get("mouthSmileRight", 0.0)
        return left > SMILE_THRESHOLD and right > SMILE_THRESHOLD


# ---- session state (resettable) -----------------------------------------


class Session:
    def __init__(self):
        self.runner = ChallengeRunner()
        self.real_count = 0
        self.total_count = 0
        self.fake_streak = 0
        self.verdict = None
        print(f"Challenge order: {self.runner.challenges}")

    def observe(self, is_real):
        self.total_count += 1
        if is_real:
            self.real_count += 1
            self.fake_streak = 0
        else:
            self.fake_streak += 1
        if self.fake_streak >= FAKE_STREAK_LIMIT:
            self.verdict = "FAKE (spoof detected)"

    def maybe_finalize(self):
        if self.verdict is None and self.runner.done and self.total_count > 0:
            real_ratio = self.real_count / self.total_count
            self.verdict = (
                "REAL & LIVE"
                if real_ratio >= REAL_RATIO_PASS
                else f"FAKE (spoof ratio too high, real_ratio={real_ratio:.2f})"
            )


# ---- main loop -----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model_dir_tflite", type=str, default=MODEL_DIR_TFLITE)
    parser.add_argument(
        "--max_seconds", type=float, default=None, help="auto-quit after N seconds (for scripted smoke tests)"
    )
    args = parser.parse_args()

    loaded_models = load_antispoof_models(args.model_dir_tflite)
    image_cropper = CropImage()
    mesh = FaceMesh()
    session = Session()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        analysis = mesh.analyze(frame)
        if analysis is None:
            cv2.putText(frame, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            bbox, shapes = analysis
            is_real, spoof_score = score_antispoof(loaded_models, image_cropper, frame, bbox)

            if session.verdict is None:
                session.observe(is_real)
                just_completed = session.runner.update(shapes)
                if just_completed:
                    print(f"Challenge completed ({session.total_count} frames so far)")
                session.maybe_finalize()

            x, y, w, h = bbox
            color = (0, 200, 0) if is_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"antispoof: {'real' if is_real else 'fake'} {spoof_score:.2f}",
                (x, max(0, y - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"challenge: {session.runner.current or 'done'}",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        if session.verdict:
            color = (0, 200, 0) if session.verdict.startswith("REAL") else (0, 0, 255)
            cv2.putText(frame, session.verdict, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, "press 'r' to retry", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Challenge + Anti-Spoofing", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            session = Session()
        if args.max_seconds and time.time() - start > args.max_seconds:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(session.verdict or "Session ended without a verdict (no face / interrupted / challenges incomplete).")


if __name__ == "__main__":
    main()
