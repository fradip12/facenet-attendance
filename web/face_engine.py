"""Face detection, blink detection, FaceNet embedding and matching."""

import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

_MODELS = Path(__file__).resolve().parent.parent / "models"
FACENET_PATH = _MODELS / "facenet.tflite"
LANDMARKER_PATH = _MODELS / "face_landmarker.task"

# eyeBlinkLeft/Right blendshape score above this reads as a closed eye
# (analog of ML Kit's eye-open probability, inverted)
BLINK_THRESHOLD = 0.5


class FaceEngine:
    """Wraps MediaPipe FaceLandmarker + the FaceNet TFLite interpreter.

    All calls are serialized behind one lock: neither MediaPipe nor the
    TFLite interpreter is thread-safe, and frames arrive from a webrtc
    worker thread while embedding runs on the main thread.
    """

    def __init__(self):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self._interpreter = Interpreter(model_path=str(FACENET_PATH))
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self.input_size = int(self._input["shape"][1])
        self.embedding_dim = int(self._output["shape"][-1])
        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
                output_face_blendshapes=True,
                num_faces=1,
            )
        )
        self._lock = threading.Lock()

    def analyze(self, bgr: np.ndarray):
        """Return (bbox, left_blink, right_blink) or None if no face found.

        blink scores are 0..1, higher = more closed.
        """
        h, w = bgr.shape[:2]
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        with self._lock:
            res = self._landmarker.detect(image)
        if not res.face_landmarks:
            return None
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bbox = (max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h))
        shapes = {s.category_name: s.score for s in res.face_blendshapes[0]}
        return bbox, shapes.get("eyeBlinkLeft", 0.0), shapes.get("eyeBlinkRight", 0.0)

    def embed(self, bgr: np.ndarray, bbox) -> np.ndarray:
        """Crop bbox, run FaceNet, return an L2-normalized embedding."""
        x1, y1, x2, y2 = bbox
        face = cv2.resize(bgr[y1:y2, x1:x2], (self.input_size, self.input_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if self._input["dtype"] == np.float32:
            face = face.astype(np.float32)
            face = (face - face.mean()) / max(float(face.std()), 1e-6)  # prewhiten
        with self._lock:
            self._interpreter.set_tensor(self._input["index"], face[None, ...])
            self._interpreter.invoke()
            emb = self._interpreter.get_tensor(self._output["index"])[0].astype(
                np.float32
            )
        return emb / (np.linalg.norm(emb) + 1e-6)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # embeddings are already L2-normalized


def best_match(embedding: np.ndarray, registry: dict):
    """registry: name -> embedding. Return (name, score) or (None, 0.0)."""
    if not registry:
        return None, 0.0
    return max(
        ((n, cosine_similarity(embedding, e)) for n, e in registry.items()),
        key=lambda t: t[1],
    )


if __name__ == "__main__":
    engine = FaceEngine()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert engine.analyze(img) is None  # noise has no face
    emb = engine.embed(img, (0, 0, 640, 480))
    assert emb.shape == (engine.embedding_dim,), emb.shape
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-3
    assert abs(cosine_similarity(emb, emb) - 1.0) < 1e-6
    name, score = best_match(emb, {"me": emb})
    assert name == "me" and score > 0.999
    assert best_match(emb, {}) == (None, 0.0)
    print(f"ok: input {engine.input_size}px, embedding dim {engine.embedding_dim}")
