import cv2
import os
import datetime
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from app.frame_extractor import extract_frames
from app.face_cropper import crop_faces
from app.autoencoder import generate_deepfakes
from app.simswap_runner import run_simswap




video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

class VideoRecorder(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        self.recording = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def record_video(user_name):
    st.subheader("Webcam Recorder")

    # Webcam stream setup (no device chooser)
    ctx = webrtc_streamer(
        key="video_capture",
        video_transformer_factory=VideoRecorder,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    # Show Start/Stop buttons once the stream is ready
    if ctx.video_transformer:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Recording"):
                ctx.video_transformer.recording = True
                st.success("Recording started...")

        with col2:
            if st.button("Stop and Save"):
                ctx.video_transformer.recording = False
                frames = ctx.video_transformer.frames

                if frames:
                    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"{video_folder}/{user_name}_{now}.avi"

                    height, width, _ = frames[0].shape
                    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 15, (width, height))
                    for frame in frames:
                        out.write(frame)
                    out.release()

                    st.success(f"🎥 Video saved as: {video_filename}")

                    # Frame extraction
                    frame_folder = f"frames/{user_name}"
                    os.makedirs(frame_folder, exist_ok=True)
                    existing_frames = [f for f in os.listdir(frame_folder) if f.endswith(".jpg")]
                    start_index = len(existing_frames)
                    frame_count = extract_frames(video_filename, frame_folder, step=5)
                    st.success(f"🖼️ Extracted {frame_count} new frames to: {frame_folder}")

                    # Face cropping
                    face_folder = f"faces/{user_name}/real"
                    os.makedirs(face_folder, exist_ok=True)
                    face_count = crop_faces(frame_folder, face_folder, start_index=start_index)
                    st.success(f"👤 Cropped {face_count} new faces to: {face_folder}")

                    # Deepfake generation (Autoencoder)
                    deepfake_folder = f"faces/{user_name}/fake"
                    generate_deepfakes(face_folder, deepfake_folder, epochs=10)
                    deepfake_count = len([f for f in os.listdir(deepfake_folder) if f.endswith(".jpg")])
                    st.success(f"🧪 Generated {deepfake_count} deepfake-style images to: {deepfake_folder}")

                    ctx.video_transformer.frames.clear()

                else:
                    st.warning("⚠️ No frames were recorded.")

