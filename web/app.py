"""Sentec face attendance web demo: register a face, verify by blinking."""

import queue
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from face_engine import BLINK_THRESHOLD, FaceEngine, best_match

st.set_page_config(page_title="Sentec Face Attendance", layout="wide")


@st.cache_resource
def get_engine() -> FaceEngine:
    return FaceEngine()


engine = get_engine()

if "faces" not in st.session_state:
    st.session_state.faces = {}  # name -> embedding
if "last_result" not in st.session_state:
    st.session_state.last_result = None


class Processor(VideoProcessorBase):
    """Runs in the webrtc worker thread; hands blink-captured frames to the UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_bbox = None
        self.blink_queue = queue.Queue(maxsize=1)
        self._last_blink = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        result = engine.analyze(img)
        with self.lock:
            self.latest_frame = img.copy()
            self.latest_bbox = result[0] if result else None
        if result:
            bbox, left_blink, right_blink = result
            closed = left_blink > BLINK_THRESHOLD and right_blink > BLINK_THRESHOLD
            color = (0, 0, 255) if closed else (0, 255, 0)
            cv2.rectangle(img, bbox[:2], bbox[2:], color, 2)
            now = time.monotonic()
            if closed and now - self._last_blink > 3.0:  # cooldown between triggers
                self._last_blink = now
                try:
                    self.blink_queue.put_nowait((img.copy(), bbox))
                except queue.Full:
                    pass
        else:
            cv2.putText(
                img, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Sentec Face Attendance")

video_col, panel_col = st.columns([2, 1])

with video_col:
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=Processor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={  # STUN needed when hosted (e.g. Streamlit Cloud)
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )
    st.caption("Green box: face detected. Red box: eyes closed → verification runs.")

with panel_col:
    st.subheader("Register")
    name = st.text_input("Name")
    if st.button("Register face", disabled=not ctx.state.playing):
        proc = ctx.video_processor
        with proc.lock:
            frame, bbox = proc.latest_frame, proc.latest_bbox
        if not name.strip():
            st.warning("Enter a name first.")
        elif frame is None or bbox is None:
            st.warning("No face in view.")
        else:
            st.session_state.faces[name.strip()] = engine.embed(frame, bbox)
            st.success(f"Registered {name.strip()}.")

    if st.session_state.faces:
        st.write("Registered:", ", ".join(st.session_state.faces))
        if st.button("Clear all"):
            st.session_state.faces = {}
            st.rerun()

    st.subheader("Verify")
    threshold = st.slider("Match threshold", 0.5, 1.0, 0.85, 0.01)
    status = st.empty()
    if st.session_state.last_result:
        status.info(st.session_state.last_result)

# Poll for blink-triggered verifications while the stream is live.
while ctx.state.playing:
    try:
        frame, bbox = ctx.video_processor.blink_queue.get(timeout=0.5)
    except queue.Empty:
        continue
    if not st.session_state.faces:
        msg = "Blink detected — register a face first."
    else:
        emb = engine.embed(frame, bbox)
        match, score = best_match(emb, st.session_state.faces)
        if score >= threshold:
            msg = f"✅ Verified: {match} (similarity {score:.2f})"
        else:
            msg = f"❌ No match (best: {match}, similarity {score:.2f})"
    st.session_state.last_result = f"{time.strftime('%H:%M:%S')} — {msg}"
    status.info(st.session_state.last_result)
