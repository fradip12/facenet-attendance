# Web Face Attendance Demo — Design

Date: 2026-07-02 · Status: approved

## Goal

A Streamlit app under `web/` mirroring the mobile attendance flow: live face
detection, blink-triggered verification, in-session face registration, using
`models/facenet.tflite` for embeddings.

## Decisions (from Q&A)

- **Detector/landmarks:** MediaPipe FaceMesh (ML Kit replacement); blink via
  eye-aspect-ratio (EAR) on eye landmarks.
- **Camera:** live stream via `streamlit-webrtc`, continuous analysis.
- **Storage:** `st.session_state` only (name → embedding), lost on refresh.

## Architecture

- `web/app.py` — UI: one live stream, register controls, verify status.
- `web/face_engine.py` — MediaPipe detect + EAR, TFLite FaceNet embedding,
  cosine matching. Model input size / embedding dim read from the interpreter
  at runtime. Embeddings L2-normalized.
- `web/requirements.txt`

## Flow

1. webrtc worker thread: per frame → FaceMesh → bbox + left/right EAR, box
   drawn on frame (red when eyes closed).
2. Register: name + button → latest frame's face cropped, prewhitened,
   embedded → stored in session state.
3. Verify: both EARs < 0.2 (with 3 s cooldown) → frame passed to main thread
   via a bounded queue → embed → cosine vs registry → match if score ≥
   threshold (slider, default 0.85).

## Error handling

No face → status message. Blink with empty registry → prompt to register.
TFLite/MediaPipe calls serialized behind a lock (not thread-safe).

## Testing

`python face_engine.py` self-check: embedding shape, unit norm,
self-similarity ≈ 1.

## Out of scope

Persistence, multi-face frames (largest/only face wins), auth, deployment.
