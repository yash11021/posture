"""
PostSURE - Real-time Posture Classification
============================================
Streamlit app for posture analysis using MediaPipe and PyTorch.
"""

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from classifier import PostureClassifier, LandmarkNormalizer


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
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")

    return model, device, model_loaded


@st.cache_resource
def load_mediapipe():
    """Load MediaPipe pose detector"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return pose, mp_pose, mp_drawing, mp_drawing_styles


def extract_landmarks(pose, frame):
    """Extract pose landmarks from a frame"""
    results = pose.process(frame)

    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        return np.array(landmarks, dtype=np.float32), results.pose_landmarks

    return None, None


def predict_posture(model, device, model_loaded, landmarks, normalizer):
    """Run inference on landmarks"""
    if landmarks is None:
        return None, None, None

    # Normalize landmarks
    try:
        normalized = normalizer.normalize(landmarks)
    except Exception:
        normalized = landmarks

    if not model_loaded:
        # Demo mode
        pred = np.random.choice([0, 1])
        conf = np.random.uniform(0.6, 0.95)
        return pred, conf, "Demo"

    with torch.no_grad():
        tensor = torch.FloatTensor(normalized).unsqueeze(0).to(device)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    return pred, conf, "Model"


def draw_results(frame, pose_landmarks, prediction, confidence, mp_pose, mp_drawing, mp_drawing_styles):
    """Draw pose and prediction results on frame"""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    # Draw pose skeleton
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Colors and text
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
    cv2.rectangle(frame_bgr, (0, 0), (w, h), border_color, 12)

    # Draw status bar
    cv2.rectangle(frame_bgr, (0, 0), (w, 60), bg_color, -1)
    cv2.putText(frame_bgr, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Confidence bar
    if confidence is not None:
        bar_width = int(150 * confidence)
        cv2.rectangle(frame_bgr, (w - 180, 20), (w - 30, 45), (50, 50, 50), -1)
        cv2.rectangle(frame_bgr, (w - 180, 20), (w - 180 + bar_width, 45), border_color, -1)
        cv2.putText(frame_bgr, f"{confidence:.0%}", (w - 175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(
        page_title="PostSURE - Posture Classifier",
        page_icon="🧘",
        layout="centered"
    )

    st.title("🧘 PostSURE - Posture Classifier")
    st.markdown("""
    Real-time posture classification using **MediaPipe** pose estimation and a **PyTorch** neural network.

    **How to use:** Take a webcam photo or upload an image to analyze your posture.
    - 🟢 **Green border** = Good posture
    - 🔴 **Red border** = Bad posture
    """)

    st.divider()

    # Load resources
    model, device, model_loaded = load_model()
    pose, mp_pose, mp_drawing, mp_drawing_styles = load_mediapipe()
    normalizer = LandmarkNormalizer()

    # Status
    if model_loaded:
        st.success("Model loaded successfully")
    else:
        st.warning("Model not found - running in demo mode with random predictions")

    # Input options
    tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload Image"])

    with tab1:
        st.markdown("Take a photo with your webcam:")
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")

        if camera_image is not None:
            # Process webcam image
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mirror the image
            frame_rgb = np.fliplr(frame_rgb).copy()

            # Analyze
            landmarks, pose_landmarks = extract_landmarks(pose, frame_rgb)
            prediction, confidence, source = predict_posture(model, device, model_loaded, landmarks, normalizer)
            result = draw_results(frame_rgb, pose_landmarks, prediction, confidence, mp_pose, mp_drawing, mp_drawing_styles)

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

    with tab2:
        st.markdown("Upload an image to analyze:")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze
            landmarks, pose_landmarks = extract_landmarks(pose, frame_rgb)
            prediction, confidence, source = predict_posture(model, device, model_loaded, landmarks, normalizer)
            result = draw_results(frame_rgb, pose_landmarks, prediction, confidence, mp_pose, mp_drawing, mp_drawing_styles)

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

        Built with PyTorch, MediaPipe, OpenCV, and Streamlit.
        """)


if __name__ == "__main__":
    main()
