#   1. Captures video from webcam                              │
# │  2. Sends each frame to MediaPipe                           │
# │  3. Extracts the 33 landmark coordinates                    │
# │  4. Feeds landmarks to PyTorch model                        │
# │  5. Gets prediction: Good (1) or Bad (0)                    │
# │  6. Smooths predictions to avoid jitter                     │
# │  7. Displays result (green/red border)


import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
import json
import os
import random
from classifier import PostureClassifier

class PostureMonitor:
    def __init__(self, model_path=None, test_mode=False):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # TEST MODE: Skip model loading entirely
        self.test_mode = test_mode
        
        if not test_mode:
            # Initialize PyTorch model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = PostureClassifier().to(self.device)
            
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.is_calibrated = True
            else:
                self.is_calibrated = False
        else:
            self.is_calibrated = True  # Pretend we're calibrated in test mode
            print("🧪 TEST MODE: Using random predictions (no model needed)")
        
        # Smoothing buffer for predictions
        self.prediction_buffer = deque(maxlen=10)
        
    def extract_landmarks(self, frame):
        """Extract pose landmarks from frame"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract x, y coordinates (ignore z for now)
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            return np.array(landmarks, dtype=np.float32), results.pose_landmarks
        
        return None, None
    
    def predict_posture(self, landmarks):
        """Predict posture from landmarks"""
        if landmarks is None:
            return None
        
        # TEST MODE: Generate random predictions
        if self.test_mode:
            print(f"📊 Landmarks shape: {landmarks.shape} (66 values)")
            
            # Generate random prediction (0 or 1)
            prediction = random.choice([0, 1])
            confidence = random.uniform(0.7, 0.95)  # Random confidence between 70-95%
            
            print(f"🎲 Random prediction: {prediction} ({'Good' if prediction == 1 else 'Bad'}) with {confidence:.1%} confidence")
            
            # Still smooth the predictions
            self.prediction_buffer.append(prediction)
            smoothed_prediction = round(np.mean(self.prediction_buffer))
            
            return smoothed_prediction, confidence
        
        # NORMAL MODE: Use the actual model
        if not self.is_calibrated:
            return None
        
        # Convert to tensor and predict
        with torch.no_grad():
            landmark_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
            print(f"📊 Tensor shape going into model: {landmark_tensor.shape}")
            
            output = self.model(landmark_tensor)
            print(f"🧠 Model raw output: {output}")
            
            probabilities = torch.softmax(output, dim=1)
            print(f"📈 Probabilities: {probabilities} (Bad={probabilities[0][0]:.1%}, Good={probabilities[0][1]:.1%})")
            
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            print(f"✅ Final prediction: {prediction} ({'Good' if prediction == 1 else 'Bad'}) with {confidence:.1%} confidence")
        
        # Smooth predictions
        self.prediction_buffer.append(prediction)
        smoothed_prediction = round(np.mean(self.prediction_buffer))
        
        return smoothed_prediction, confidence
    
    def run(self):
        """Main monitoring loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("Posture Monitor Started!")
        print("="*60)
        
        if self.test_mode:
            print("🧪 TEST MODE ACTIVE")
            print("   - MediaPipe will detect your pose")
            print("   - Predictions are RANDOM (for demo)")
            print("   - Watch the console to see the data flow!")
        elif not self.is_calibrated:
            print("⚠️  Model not calibrated. Please run calibration first.")
        else:
            print("✓ Model loaded and ready")
        
        print("\nPress 'q' to quit")
        print("="*60 + "\n")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            print(f"\n--- Frame {frame_count} ---")
            print("1️⃣ Capturing webcam frame...")
            
            landmarks, pose_landmarks = self.extract_landmarks(frame)
            
            if landmarks is not None:
                print(f"2️⃣ MediaPipe detected pose! Found {len(landmarks)//2} landmarks")
            else:
                print("❌ No pose detected in this frame")
            
            # Default border color
            border_color = (128, 128, 128)  # Gray for uncalibrated
            status_text = "Not Calibrated"
            
            if self.is_calibrated and landmarks is not None:
                # Predict posture
                print("3️⃣ Feeding landmarks to model...")
                result = self.predict_posture(landmarks)
                
                if result:
                    prediction, confidence = result
                    print(f"4️⃣ Smoothed prediction: {prediction} ({'Good' if prediction == 1 else 'Bad'})")
                    
                    if prediction == 1:  # Good posture
                        border_color = (0, 255, 0)  # Green
                        status_text = f"Good Posture ({confidence:.1%})"
                        print("5️⃣ Displaying: GREEN BORDER ✓")
                    else:  # Bad posture
                        border_color = (0, 0, 255)  # Red
                        status_text = f"Bad Posture ({confidence:.1%})"
                        print("5️⃣ Displaying: RED BORDER ✗")
            
            # Draw pose landmarks
            if pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            # Draw border
            border_thickness = 15
            cv2.rectangle(
                frame, 
                (0, 0), 
                (frame.shape[1], frame.shape[0]), 
                border_color, 
                border_thickness
            )
            
            # Draw status text
            cv2.putText(
                frame, 
                status_text, 
                (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                border_color, 
                2
            )
            
            # Add test mode indicator
            if self.test_mode:
                cv2.putText(
                    frame,
                    "TEST MODE - Random Predictions",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
            
            cv2.imshow('Posture Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    import sys
    
    # Check if user wants test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n🧪 Starting in TEST MODE")
        print("This will show you how the system works WITHOUT training a model\n")
        monitor = PostureMonitor(test_mode=True)
    else:
        print("\n💡 TIP: Run with --test flag to test without a trained model:")
        print("   python monitor.py --test\n")
        monitor = PostureMonitor(model_path='models/posture_model.pth')
    
    monitor.run()