"""
Real-time Posture Classification Demo
=====================================
A PyTorch-based posture classifier using MediaPipe pose estimation.

This demo showcases:
- MediaPipe pose landmark extraction
- Custom PyTorch neural network with residual connections
- Real-time webcam inference with Gradio
- Audio alerts for bad posture

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
import time

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
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                self.model_loaded = True
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Running in demo mode with random predictions")

        # Smoothing buffer
        self.prediction_buffer = deque(maxlen=5)

        # Alert state tracking
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # seconds between alerts
        self.consecutive_bad = 0
        self.bad_threshold = 3  # consecutive bad frames before alert

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

    def should_alert(self, prediction):
        """Check if we should trigger an alert"""
        current_time = time.time()

        if prediction == 0:  # Bad posture
            self.consecutive_bad += 1
        else:
            self.consecutive_bad = 0
            return False

        # Alert if consecutive bad frames exceed threshold and cooldown passed
        if (self.consecutive_bad >= self.bad_threshold and
            current_time - self.last_alert_time > self.alert_cooldown):
            self.last_alert_time = current_time
            self.consecutive_bad = 0  # Reset after alert
            return True

        return False

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
            status = "Good Posture"
            bg_color = (0, 120, 0)
        else:
            border_color = (0, 0, 200)  # Red
            status = "Bad Posture - Sit up!"
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
            return None, False

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Extract and predict
        landmarks, pose_landmarks = self.extract_landmarks(frame)
        prediction, confidence, source = self.predict(landmarks)

        # Check if alert needed
        trigger_alert = self.should_alert(prediction)

        # Draw results
        result = self.draw_results(frame, pose_landmarks, prediction, confidence, source)

        return result, trigger_alert


# Initialize demo
demo_instance = PostureDemo(model_path="models/posture_model.pth", use_improved=False)


def process_realtime(frame):
    """Process frame and return result with alert status"""
    if frame is None:
        return None, ""

    # Gradio provides RGB numpy array, convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result, trigger_alert = demo_instance.process_frame(frame_bgr)

    # Convert back to RGB for Gradio display
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Return alert status for JavaScript to handle
    alert_status = "ALERT" if trigger_alert else ""

    return result_rgb, alert_status


def process_image(image):
    """Process a single uploaded image"""
    if image is None:
        return None
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result, _ = demo_instance.process_frame(frame_bgr)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# JavaScript for audio alert
ALERT_JS = """
async () => {
    // Create audio context for alert sound
    window.postureAudioCtx = window.postureAudioCtx || new (window.AudioContext || window.webkitAudioContext)();

    window.playPostureAlert = function() {
        const ctx = window.postureAudioCtx;
        const oscillator = ctx.createOscillator();
        const gainNode = ctx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(ctx.destination);

        oscillator.frequency.value = 440;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);

        oscillator.start(ctx.currentTime);
        oscillator.stop(ctx.currentTime + 0.5);
    };

    // Watch for alert status changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.target.textContent === 'ALERT') {
                window.playPostureAlert();
            }
        });
    });

    // Start observing after a short delay to let Gradio render
    setTimeout(() => {
        const alertBox = document.querySelector('#alert-status textarea');
        if (alertBox) {
            observer.observe(alertBox, { childList: true, characterData: true, subtree: true });
        }
    }, 2000);
}
"""

# Build Gradio interface
with gr.Blocks(
    title="PostSURE - Posture Monitor",
    theme=gr.themes.Soft(),
    js=ALERT_JS,
    css="""
    .gradio-container { max-width: 1000px !important; }
    #alert-status { display: none; }
    """
) as app:
    gr.Markdown(
        """
        # PostSURE - Real-time Posture Monitor

        This demo uses **MediaPipe** for pose estimation and a **PyTorch neural network**
        for posture classification. Keep this tab open and it will alert you when your posture is bad!

        ### How to use
        1. Allow camera access
        2. Click **Start Monitoring**
        3. Work normally - you'll hear a beep when posture is bad

        ---
        """
    )

    # Hidden alert status for JS to monitor
    alert_status = gr.Textbox(value="", elem_id="alert-status", visible=False)

    with gr.Tabs():
        with gr.TabItem("Live Monitor"):
            gr.Markdown("Position yourself in frame. **Green** = good posture, **Red** = bad posture (will beep!)")

            with gr.Row():
                webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Your Camera")
                webcam_output = gr.Image(label="Analysis")

            webcam_input.stream(
                fn=process_realtime,
                inputs=webcam_input,
                outputs=[webcam_output, alert_status],
                stream_every=0.1
            )

        with gr.TabItem("Upload Image"):
            gr.Markdown("Upload an image to analyze posture (no alerts).")
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
        - Input: 66 features (33 MediaPipe landmarks x 2 coordinates)
        - Residual blocks with batch normalization
        - Landmark normalization for position/scale invariance

        **Alert System:**
        - Triggers after 3 consecutive "bad posture" frames
        - 3-second cooldown between alerts
        - Audio beep when tab is active

        Built with PyTorch, MediaPipe, and Gradio.
        """
    )


if __name__ == "__main__":
    app.launch()
