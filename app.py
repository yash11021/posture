"""
PostSURE - Real-time Posture Classification
============================================
Streamlit app for posture analysis using MediaPipe and PyTorch.
Uses WebRTC for real-time browser-based webcam streaming.
"""

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from classifier import PostureClassifier, LandmarkNormalizer


# WebRTC configuration for STUN server (needed for NAT traversal)
# Using Google's public STUN server which is reliable for most networks
# WebRTC configuration
# Using a free open relay list which is often more reliable than just Google's STUN
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)


@st.cache_resource
def load_model(model_path="models/posture_model.pth"):
    """Load model once and cache it"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PostureClassifier(input_size=66).to(device)

    model_loaded = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            model_loaded = True
        except Exception as e:
            print(f"Could not load model: {e}")

    return model, device, model_loaded


def create_pose_detector():
    """Create a new MediaPipe pose detector instance (not cached - thread-safe)"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return pose, mp_pose, mp_drawing, mp_drawing_styles


class VideoProcessor:
    """WebRTC video processor for real-time posture analysis"""

    def __init__(self):
        # Load model (cached, thread-safe for inference)
        self.model, self.device, self.model_loaded = load_model()

        # Create fresh MediaPipe instance for this processor (not cached - MediaPipe is not thread-safe)
        self.pose, self.mp_pose, self.mp_drawing, self.mp_drawing_styles = create_pose_detector()
        self.normalizer = LandmarkNormalizer()

        # Store latest results for display
        self.latest_prediction = None
        self.latest_confidence = None

    def recv(self, frame):
        """Process each video frame"""
        img = frame.to_ndarray(format="bgr24")

        # Mirror the image for natural interaction
        img = cv2.flip(img, 1)

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract landmarks
        results = self.pose.process(img_rgb)

        prediction = None
        confidence = None

        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Extract landmark coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            landmarks = np.array(landmarks, dtype=np.float32)

            # Normalize
            try:
                normalized = self.normalizer.normalize(landmarks)
            except Exception:
                normalized = landmarks

            # Predict
            if self.model_loaded:
                with torch.no_grad():
                    tensor = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1)
                    prediction = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][prediction].item()
            else:
                # Demo mode
                prediction = np.random.choice([0, 1])
                confidence = np.random.uniform(0.6, 0.95)

        # Store for external access
        self.latest_prediction = prediction
        self.latest_confidence = confidence

        # Draw results on frame
        h, w = img.shape[:2]

        # Colors based on prediction
        if prediction is None:
            border_color = (128, 128, 128)
            status = "No Pose Detected"
            bg_color = (80, 80, 80)
        elif prediction == 1:
            border_color = (0, 200, 0)
            status = "Good Posture"
            bg_color = (0, 120, 0)
        else:
            border_color = (0, 0, 200)
            status = "Bad Posture - Sit up!"
            bg_color = (0, 0, 120)

        # Draw border
        cv2.rectangle(img, (0, 0), (w, h), border_color, 12)

        # Draw status bar
        cv2.rectangle(img, (0, 0), (w, 50), bg_color, -1)
        cv2.putText(img, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Confidence bar
        if confidence is not None:
            bar_width = int(120 * confidence)
            cv2.rectangle(img, (w - 150, 15), (w - 30, 40), (50, 50, 50), -1)
            cv2.rectangle(img, (w - 150, 15), (w - 150 + bar_width, 40), border_color, -1)
            cv2.putText(img, f"{confidence:.0%}", (w - 145, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def process_image(frame_rgb, model, device, model_loaded, pose, mp_pose, mp_drawing, mp_drawing_styles, normalizer):
    """Process a single image and return results"""
    results = pose.process(frame_rgb)

    prediction = None
    confidence = None
    source = None

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        landmarks = np.array(landmarks, dtype=np.float32)

        # Normalize
        try:
            normalized = normalizer.normalize(landmarks)
        except Exception:
            normalized = landmarks

        # Predict
        if model_loaded:
            with torch.no_grad():
                tensor = torch.FloatTensor(normalized).unsqueeze(0).to(device)
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
                source = "Model"
        else:
            prediction = np.random.choice([0, 1])
            confidence = np.random.uniform(0.6, 0.95)
            source = "Demo"

    # Draw results
    if prediction is None:
        border_color = (128, 128, 128)
        status = "No Pose Detected"
        bg_color = (80, 80, 80)
    elif prediction == 1:
        border_color = (0, 200, 0)
        status = "Good Posture"
        bg_color = (0, 120, 0)
    else:
        border_color = (0, 0, 200)
        status = "Bad Posture - Sit up!"
        bg_color = (0, 0, 120)

    cv2.rectangle(frame_bgr, (0, 0), (w, h), border_color, 12)
    cv2.rectangle(frame_bgr, (0, 0), (w, 60), bg_color, -1)
    cv2.putText(frame_bgr, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    if confidence is not None:
        bar_width = int(150 * confidence)
        cv2.rectangle(frame_bgr, (w - 180, 20), (w - 30, 45), (50, 50, 50), -1)
        cv2.rectangle(frame_bgr, (w - 180, 20), (w - 180 + bar_width, 45), border_color, -1)
        cv2.putText(frame_bgr, f"{confidence:.0%}", (w - 175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb, prediction, confidence, source


def main():
    st.set_page_config(
        page_title="PostSURE - Posture Classifier",
        page_icon="🧘",
        layout="centered"
    )

    st.title("🧘 PostSURE - Posture Classifier")
    st.markdown("""
    Real-time posture classification using **MediaPipe** pose estimation and a **PyTorch** neural network.

    - 🟢 **Green border** = Good posture
    - 🔴 **Red border** = Bad posture
    """)

    # Load resources for status display
    model, device, model_loaded = load_model()

    # Status indicator
    if model_loaded:
        st.success("Model loaded successfully")
    else:
        st.warning("Model not found - running in demo mode with random predictions")

    st.divider()

    # Mode selection
    mode = st.radio(
        "Select mode:",
        ["Real-time Webcam", "Upload Image"],
        horizontal=True
    )

    if mode == "Real-time Webcam":
        st.markdown("### Real-time Posture Monitoring")
        st.markdown("Click **START** to begin real-time posture analysis from your webcam.")

        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="posture-monitor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.info("Monitoring active - sit naturally and watch your posture!")
        else:
            st.markdown("*Click START above to begin monitoring*")

    else:  # Upload Image
        st.markdown("### Upload an Image")
        st.markdown("Upload an image to analyze posture:")

        # Create fresh pose detector for image processing
        pose, mp_pose, mp_drawing, mp_drawing_styles = create_pose_detector()
        normalizer = LandmarkNormalizer()

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process
            result, prediction, confidence, source = process_image(
                frame_rgb, model, device, model_loaded,
                pose, mp_pose, mp_drawing, mp_drawing_styles, normalizer
            )

            st.image(result, caption="Analysis Result", use_container_width=True)

            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if prediction == 1:
                    st.metric("Status", "Good", delta="Posture OK")
                elif prediction == 0:
                    st.metric("Status", "Bad", delta="Needs correction", delta_color="inverse")
                else:
                    st.metric("Status", "Unknown", delta="No pose detected")
            with col2:
                if confidence:
                    st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("Source", source if source else "N/A")

    # Technical details
    st.divider()
    with st.expander("Technical Details"):
        st.markdown("""
        ### Model Architecture
        - **Input**: 66 features (33 MediaPipe landmarks × 2 coordinates)
        - **Architecture**: Feedforward neural network with dropout regularization
        - **Preprocessing**: Landmark normalization (centered at hips, scaled by shoulder width)

        ### Pipeline
        1. **MediaPipe Pose** extracts 33 body keypoints
        2. **Normalizer** transforms coordinates to be position/scale invariant
        3. **PyTorch model** classifies posture as good or bad
        4. **Visualization** overlays skeleton and prediction

        ### Real-time Streaming
        Uses WebRTC for low-latency video streaming from your browser to the server.

        Built with PyTorch, MediaPipe, OpenCV, Streamlit, and streamlit-webrtc.
        """)


if __name__ == "__main__":
    main()
