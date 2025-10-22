import sys
import os
import mediapipe as mp
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import datetime

from classifier import PostureClassifier


class PostureCalibrator:
    """Collect calibration data and train the model with data accumulation"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        self.data_file = 'data/training_samples.pkl'
        self.model_dir = 'models'
    
    def save_samples(self, good_samples, bad_samples):
        """
        Save feature arrays to disk
        Only stores 66 floats per sample (very small!)
        """
        data = {
            'good_samples': good_samples,
            'bad_samples': bad_samples,
            'timestamp': datetime.now().isoformat(),
            'num_good': len(good_samples),
            'num_bad': len(bad_samples)
        }
        
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Calculate size
        file_size = os.path.getsize(self.data_file)
        print(f"\n💾 Saved training data:")
        print(f"   File: {self.data_file}")
        print(f"   Size: {file_size / 1024:.2f} KB")
        print(f"   Good samples: {len(good_samples)}")
        print(f"   Bad samples: {len(bad_samples)}")
    
    def load_samples(self):
        """Load existing training samples from disk"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n📂 Loaded existing training data:")
            print(f"   File: {self.data_file}")
            print(f"   Saved: {data.get('timestamp', 'unknown')}")
            print(f"   Good samples: {len(data['good_samples'])}")
            print(f"   Bad samples: {len(data['bad_samples'])}")
            
            return data['good_samples'], data['bad_samples']
        else:
            print(f"\n📂 No existing data found at {self.data_file}")
            return [], []
    
    def collect_samples(self, label, num_samples=30, delay_frames=30):
        """
        Collect samples for a specific posture label
        
        Args:
            label: 1 for good posture, 0 for bad posture
            num_samples: Number of samples to collect
            delay_frames: Frames to wait between samples (~30 = 1 second)
        """
        cap = cv2.VideoCapture(0)
        samples = []
        
        label_name = "GOOD POSTURE" if label == 1 else "BAD POSTURE"
        
        fps = 30
        delay_seconds = delay_frames / fps
        
        print(f"\n📸 Collecting {num_samples} samples for {label_name}")
        print(f"⏱️  Delay between samples: ~{delay_seconds:.1f} seconds")
        print("Position yourself and press SPACE to start")
        print("💡 Move slightly between samples for variety!")

        collecting = False
        collected_count = 0
        frame_count = 0
        
        while cap.isOpened() and collected_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks (this is what we save - just 66 numbers!)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                # Draw pose
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                if collecting:
                    # Collect sample every 'delay_frames' frames
                    if frame_count >= delay_frames:
                        samples.append(landmarks)  # Save 66 features
                        collected_count += 1
                        frame_count = 0
                        
                        # Green flash
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                    (0, 255, 0), 20)
                    
                    frame_count += 1
                    
                    # Countdown
                    frames_remaining = delay_frames - frame_count
                    if frames_remaining > 0:
                        cv2.putText(
                            frame, 
                            f"Next in: {frames_remaining} frames", 
                            (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (255, 255, 0), 
                            2
                        )
                    
                    cv2.putText(
                        frame, 
                        f"Collected: {collected_count}/{num_samples}", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                else:
                    cv2.putText(
                        frame, 
                        f"Press SPACE - {label_name}", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 0), 
                        2
                    )
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not collecting:
                collecting = True
                frame_count = delay_frames  # First sample immediately
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Collected {len(samples)} samples")
        return samples
    
    def train(self, X, y, num_epochs=100, batch_size=8, learning_rate=0.001,
              print_every=10, model_name='posture_model.pth'):
        """Train the model on provided data"""
        
        model_path = os.path.join(self.model_dir, model_name)
        
        print("\n🔧 Training model:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Training samples: {len(X)}")
        print()
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PostureClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            if (epoch + 1) % print_every == 0 or epoch == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        final_accuracy = 100 * correct / total
        print(f"\n✓ Training complete!")
        print(f"✓ Final accuracy: {final_accuracy:.2f}%")
        print(f"✓ Best accuracy: {best_accuracy:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved: {model_path}")
        
        return model
    
    def run_calibration(self, mode='fresh', num_samples=50, delay_frames=30,
                       num_epochs=100, batch_size=8, learning_rate=0.001):
        """
        Main calibration flow
        
        Args:
            mode: 'fresh' (new data only) or 'append' (add to existing)
            num_samples: Samples to collect per posture type
            delay_frames: Delay between samples
            num_epochs: Training epochs
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
        """
        print("="*60)
        print("POSTURE CALIBRATION")
        print("="*60)
        print(f"Mode: {mode.upper()}")
        print("="*60)
        
        # Load existing data if appending
        existing_good = []
        existing_bad = []
        
        if mode == 'append':
            existing_good, existing_bad = self.load_samples()
            if existing_good or existing_bad:
                print(f"\n✓ Will ADD to existing data")
            else:
                print(f"\n⚠️  No existing data - starting fresh")
        
        # Collect good posture
        print("\n" + "="*60)
        print("STEP 1: GOOD POSTURE")
        print("="*60)
        print("Tips:")
        print("  • Sit up VERY straight")
        print("  • Shoulders back and down")
        print("  • Head aligned with spine")
        print("  • Move slightly between samples")
        
        new_good = self.collect_samples(label=1, num_samples=num_samples, 
                                        delay_frames=delay_frames)
        
        # Collect bad posture
        print("\n" + "="*60)
        print("STEP 2: BAD POSTURE")
        print("="*60)
        print("Tips:")
        print("  • Slouch forward DRAMATICALLY")
        print("  • Round your shoulders")
        print("  • Lean head forward")
        print("  • Move slightly between samples")
        
        new_bad = self.collect_samples(label=0, num_samples=num_samples,
                                       delay_frames=delay_frames)
        
        # Combine data
        if mode == 'append':
            all_good = existing_good + new_good
            all_bad = existing_bad + new_bad
            
            print(f"\n📊 COMBINED DATA:")
            print(f"   Previous: {len(existing_good)} good, {len(existing_bad)} bad")
            print(f"   New: {len(new_good)} good, {len(new_bad)} bad")
            print(f"   TOTAL: {len(all_good)} good, {len(all_bad)} bad")
        else:
            all_good = new_good
            all_bad = new_bad
            
            print(f"\n📊 NEW DATA:")
            print(f"   {len(all_good)} good samples")
            print(f"   {len(all_bad)} bad samples")
        
        # Save samples to disk
        self.save_samples(all_good, all_bad)
        
        # Prepare training data
        X = np.array(all_good + all_bad, dtype=np.float32)
        y = np.array([1]*len(all_good) + [0]*len(all_bad), dtype=np.int64)
        
        print(f"\n📊 Training dataset:")
        print(f"   Total samples: {len(X)}")
        print(f"   Features per sample: {X.shape[1]} (33 landmarks × 2 coords)")
        print(f"   Good posture: {np.sum(y == 1)}")
        print(f"   Bad posture: {np.sum(y == 0)}")
        
        # Train model
        model = self.train(X, y, num_epochs=num_epochs, batch_size=batch_size,
                          learning_rate=learning_rate)
        
        print("\n" + "="*60)
        print("✅ CALIBRATION COMPLETE!")
        print("="*60)
        print(f"Data saved: {self.data_file}")
        print(f"Model saved: {os.path.join(self.model_dir, 'posture_model.pth')}")
        print("\nNext: Run the monitor")
        print("  python monitor.py")
        print("="*60)
        
        return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("POSTURE MONITOR - CALIBRATION")
    print("="*60)
    print("\nChoose mode:")
    print("  1 - FRESH START (discard old data, collect new)")
    print("  2 - APPEND (add to existing data)")
    print("  3 - RETRAIN ONLY (use existing data, no collection)")
    print("="*60)
    
    choice = input("\nEnter 1, 2, or 3: ").strip()
    
    calibrator = PostureCalibrator()
    
    if choice == '1':
        # Fresh start
        print("\n🆕 Starting fresh (old data will be replaced)")
        calibrator.run_calibration(
            mode='fresh',
            num_samples=30,
            delay_frames=10,
            num_epochs=100,
            batch_size=8,
            learning_rate=0.001
        )
        
    elif choice == '2':
        # Append to existing
        print("\n➕ Appending to existing data")
        calibrator.run_calibration(
            mode='append',
            num_samples=30,  # Add 30 more of each
            delay_frames=10,
            num_epochs=100,
            batch_size=8,
            learning_rate=0.001
        )
        
    elif choice == '3':
        # Just retrain on existing data
        print("\n🔄 Retraining on existing data")
        
        good_samples, bad_samples = calibrator.load_samples()
        
        if not good_samples or not bad_samples:
            print("❌ No existing training data found!")
            print(f"   Looking for: {calibrator.data_file}")
            sys.exit(1)
        
        X = np.array(good_samples + bad_samples, dtype=np.float32)
        y = np.array([1]*len(good_samples) + [0]*len(bad_samples), dtype=np.int64)
        
        calibrator.train(X, y, num_epochs=150, batch_size=8, learning_rate=0.0005)
        
        print("\n✅ Retraining complete!")
        
    else:
        print("❌ Invalid choice!")
        sys.exit(1)