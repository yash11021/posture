"""
Real-time Posture Classification Demo
=====================================
A PyTorch-based posture classifier using MediaPipe pose estimation.

This demo showcases:
- MediaPipe pose landmark extraction
- Custom PyTorch neural network with residual connections
- Real-time webcam inference with Gradio

Author: [Your Name]
"""

import gradio as gr
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from classifier import PostureClassifier, ImprovedPostureClassifier, LandmarkNormalizer


class PostureDemo:
    """Gradio-compatible posture classification demo"""

    def __init__(self, model_path="models/posture_model.pth", use_improved=False):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_improved = use_improved
        self.normalizer = LandmarkNormalizer()

        if use_improved:
            self.model = ImprovedPostureClassifier(input_size=66, num_classes=2).to(self.device)
        else:
            self.model = PostureClassifier(input_size=66).to(self.device)

        # Load weights if available
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Running in demo mode with random predictions")

        # Smoothing buffer
        self.prediction_buffer = deque(maxlen=5)

    def extract_landmarks(self, frame):
        """Extract pose landmarks from a frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            return np.array(landmarks, dtype=np.float32), results.pose_landmarks

        return None, None

    def predict(self, landmarks):
        """Run inference on landmarks"""
        if landmarks is None:
            return None, None, None

        # Normalize landmarks for better generalization
        try:
            normalized = self.normalizer.normalize(landmarks)
        except Exception:
            normalized = landmarks

        if not self.model_loaded:
            # Demo mode: return random prediction
            pred = np.random.choice([0, 1])
            conf = np.random.uniform(0.6, 0.95)
            return pred, conf, "Demo Mode"

        with torch.no_grad():
            tensor = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        # Smooth predictions
        self.prediction_buffer.append(pred)
        smoothed = round(np.mean(self.prediction_buffer))

        return smoothed, conf, "Model"

    def draw_results(self, frame, pose_landmarks, prediction, confidence, source):
        """Draw pose and prediction results on frame"""
        h, w = frame.shape[:2]

        # Draw pose skeleton with custom style
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Determine colors and text based on prediction
        if prediction is None:
            border_color = (128, 128, 128)  # Gray
            status = "No Pose Detected"
            bg_color = (80, 80, 80)
        elif prediction == 1:
            border_color = (0, 200, 0)  # Green
            status = f"Good Posture"
            bg_color = (0, 120, 0)
        else:
            border_color = (0, 0, 200)  # Red
            status = f"Bad Posture"
            bg_color = (0, 0, 120)

        # Draw border
        thickness = 12
        cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)

        # Draw status bar at top
        cv2.rectangle(frame, (0, 0), (w, 60), bg_color, -1)

        # Status text
        cv2.putText(frame, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Confidence bar
        if confidence is not None:
            bar_width = int(150 * confidence)
            cv2.rectangle(frame, (w - 180, 20), (w - 30, 45), (50, 50, 50), -1)
            cv2.rectangle(frame, (w - 180, 20), (w - 180 + bar_width, 45), border_color, -1)
            cv2.putText(frame, f"{confidence:.0%}", (w - 175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Source indicator
        if source:
            cv2.putText(frame, f"[{source}]", (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def process_frame(self, frame):
        """Process a single frame for Gradio"""
        if frame is None:
            return None

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Extract and predict
        landmarks, pose_landmarks = self.extract_landmarks(frame)
        prediction, confidence, source = self.predict(landmarks)

        # Draw results
        result = self.draw_results(frame, pose_landmarks, prediction, confidence, source)

        return result


# Initialize demo
demo_instance = PostureDemo(model_path="models/posture_model.pth", use_improved=False)


def process_webcam(frame):
    """Gradio interface function"""
    return demo_instance.process_frame(frame)


def process_image(image):
    """Process a single uploaded image"""
    if image is None:
        return None
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = demo_instance.process_frame(frame)
    # Convert back to RGB for Gradio
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# Build Gradio interface
with gr.Blocks(
    title="Posture Classifier",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 900px !important; }
    .gr-button { min-width: 120px; }
    """
) as app:
    gr.Markdown(
        """
        # Real-time Posture Classification

        This demo uses **MediaPipe** for pose estimation and a **PyTorch neural network**
        for posture classification. The model analyzes 33 body landmarks to determine
        if your posture is good or bad.

        ### Features
        - **MediaPipe Pose**: Extracts 33 body keypoints in real-time
        - **PyTorch Model**: Custom neural network with batch normalization
        - **Landmark Normalization**: Position and scale invariant features
        - **Temporal Smoothing**: Reduces prediction jitter

        ---
        """
    )

    with gr.Tabs():
        with gr.TabItem("Webcam"):
            gr.Markdown("**Allow camera access** and position yourself in frame. Sit up straight for 'Good Posture'.")
            webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
            webcam_output = gr.Image(label="Analysis")
            webcam.stream(fn=process_webcam, inputs=webcam, outputs=webcam_output)

        with gr.TabItem("Upload Image"):
            gr.Markdown("Upload an image to analyze posture.")
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Upload Image")
                image_output = gr.Image(label="Analysis")
            analyze_btn = gr.Button("Analyze Posture", variant="primary")
            analyze_btn.click(fn=process_image, inputs=image_input, outputs=image_output)

    gr.Markdown(
        """
        ---
        ### Technical Details

        **Model Architecture:**
        ```
        Input (66 features: 33 landmarks x 2 coords)
            ↓
        BatchNorm → ResidualBlock(128) → ResidualBlock(64) → ResidualBlock(32)
            ↓
        Linear(16) → ReLU → Dropout → Linear(2) → Softmax
        ```

        **Landmark Normalization:**
        - Centered at hip midpoint (translation invariant)
        - Scaled by shoulder width (scale invariant)

        Built with PyTorch, MediaPipe, and Gradio.
        """
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", show_api=False)
