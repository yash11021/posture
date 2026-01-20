# PostSURE Architecture Guide

Internal documentation for understanding the codebase, data flow, and model architecture.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              POSTSURE PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WEBCAM ──► MEDIAPIPE ──► NORMALIZER ──► PYTORCH MODEL ──► PREDICTION     │
│     │          │              │               │                │            │
│   frame    33 landmarks    66 floats       logits         good/bad         │
│  (H,W,3)    (x,y) pairs    normalized     [2 classes]     + confidence     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure & Responsibilities

```
posture/
├── app.py                    # Gradio web interface (HuggingFace deployment)
├── src/
│   ├── classifier.py         # Model definitions + normalization
│   ├── calibrator.py         # Data collection + training pipeline
│   └── monitor.py            # CLI real-time monitor (OpenCV)
├── models/
│   └── posture_model.pth     # Trained weights (state_dict)
└── data/
    └── training_samples.pkl  # Collected landmark samples
```

---

## Core Classes

### 1. `PostureClassifier` (src/classifier.py:6)

**Purpose:** Original simple MLP model (legacy, still used by default)

```python
PostureClassifier(input_size=66)
```

**Architecture:**
```
Input: 66 features (33 landmarks × 2 coords)
    │
    ▼
Linear(66 → 128) + ReLU + Dropout(0.3)
    │
    ▼
Linear(128 → 64) + ReLU + Dropout(0.3)
    │
    ▼
Linear(64 → 32) + ReLU
    │
    ▼
Linear(32 → 2)  ──► Output: [logit_bad, logit_good]
```

**Parameters:** ~18,978 trainable
- Layer 1: 66×128 + 128 = 8,576
- Layer 2: 128×64 + 64 = 8,256
- Layer 3: 64×32 + 32 = 2,080
- Layer 4: 32×2 + 2 = 66

**Limitations:**
- No batch normalization (training can be unstable)
- No skip connections (gradient flow issues in deeper variants)
- No input normalization (relies on external normalizer)

---

### 2. `ResidualBlock` (src/classifier.py:26)

**Purpose:** Building block for ImprovedPostureClassifier with skip connections

```python
ResidualBlock(in_features, out_features, dropout=0.3)
```

**Architecture:**
```
Input (in_features)
    │
    ├────────────────────────────┐
    │                            │ (shortcut)
    ▼                            │
Linear(in → out)                 │
    │                            │
BatchNorm1d(out)                 │
    │                            │
ReLU                             │
    │                            │
Dropout(0.3)                     │
    │                            │
Linear(out → out)                │
    │                            │
BatchNorm1d(out)                 │
    │                            │
    ▼                            ▼
    └──────── ADD ◄──────────────┘
              │
              ▼
            ReLU
              │
              ▼
         Output (out_features)
```

**Key Points:**
- If `in_features != out_features`, shortcut uses a Linear projection
- BatchNorm after each Linear (pre-activation pattern variant)
- Dropout only in main path, not shortcut

---

### 3. `ImprovedPostureClassifier` (src/classifier.py:49)

**Purpose:** Enhanced model with modern architecture patterns

```python
ImprovedPostureClassifier(
    input_size=66,
    num_classes=2,
    hidden_dims=[128, 64, 32],  # Configurable!
    dropout=0.3
)
```

**Architecture:**
```
Input: 66 features
    │
    ▼
BatchNorm1d(66)  ◄── Input normalization
    │
    ▼
ResidualBlock(66 → 128)
    │
    ▼
ResidualBlock(128 → 64)
    │
    ▼
ResidualBlock(64 → 32)
    │
    ▼
Linear(32 → 16) + ReLU + Dropout(0.15)
    │
    ▼
Linear(16 → 2)  ──► Output: [logit_bad, logit_good]
```

**Parameters:** ~25,000+ trainable (varies with hidden_dims)

**Advantages over PostureClassifier:**
- BatchNorm stabilizes training
- Residual connections help gradient flow
- Configurable depth via `hidden_dims`
- Built-in input normalization layer

**Tuning Parameters:**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `hidden_dims` | [128, 64, 32] | More/larger = more capacity, risk overfitting |
| `dropout` | 0.3 | Higher = more regularization |
| `num_classes` | 2 | Could extend to multi-class posture types |

---

### 4. `LandmarkNormalizer` (src/classifier.py:100)

**Purpose:** Transform raw MediaPipe coordinates to position/scale invariant features

```python
LandmarkNormalizer.normalize(landmarks: np.ndarray) -> np.ndarray
```

**Why Normalize?**

Raw MediaPipe landmarks are **position-dependent**:
```
Same posture, different positions in frame:
- Sitting left:  shoulder_x = 0.2
- Sitting right: shoulder_x = 0.8
- Sitting close: shoulder_width = 0.4
- Sitting far:   shoulder_width = 0.1
```

The model would learn these as DIFFERENT inputs!

**Normalization Steps:**

```
1. FIND REFERENCE POINTS
   ┌─────────────────────────────────────┐
   │  left_hip = landmarks[23]           │
   │  right_hip = landmarks[24]          │
   │  hip_center = (left + right) / 2    │  ◄── ORIGIN
   │                                     │
   │  left_shoulder = landmarks[11]      │
   │  right_shoulder = landmarks[12]     │
   │  shoulder_width = distance(L, R)    │  ◄── SCALE
   └─────────────────────────────────────┘

2. TRANSFORM ALL POINTS
   ┌─────────────────────────────────────┐
   │  for each landmark:                 │
   │    normalized = (point - hip_center)│
   │                 ─────────────────── │
   │                   shoulder_width    │
   └─────────────────────────────────────┘
```

**Result:**
- All landmarks centered at (0, 0) = hip center
- Scale normalized so shoulder width ≈ 1.0
- Same posture → same normalized values regardless of position/distance

**Key Landmark Indices (MediaPipe Pose):**
```
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
```

---

### 5. `PostureCalibrator` (src/calibrator.py:14)

**Purpose:** Data collection and model training pipeline

```python
calibrator = PostureCalibrator()
calibrator.run_calibration(mode='fresh', num_samples=30, ...)
```

**Data Collection Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│                    CALIBRATION FLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. COLLECT GOOD POSTURE                                    │
│     ┌─────────────────────────────────────────────┐        │
│     │ User sits up straight                        │        │
│     │ MediaPipe extracts landmarks                 │        │
│     │ Save 66-float array every N frames          │        │
│     │ Repeat for num_samples (default: 30)        │        │
│     └─────────────────────────────────────────────┘        │
│                         │                                   │
│                         ▼                                   │
│  2. COLLECT BAD POSTURE                                     │
│     ┌─────────────────────────────────────────────┐        │
│     │ User slouches dramatically                   │        │
│     │ Same process as above                        │        │
│     └─────────────────────────────────────────────┘        │
│                         │                                   │
│                         ▼                                   │
│  3. SAVE TO DISK                                            │
│     data/training_samples.pkl                               │
│     {                                                       │
│       'good_samples': [[66 floats], [66 floats], ...],     │
│       'bad_samples': [[66 floats], [66 floats], ...],      │
│       'timestamp': '2024-...',                             │
│     }                                                       │
│                         │                                   │
│                         ▼                                   │
│  4. TRAIN MODEL                                             │
│     - Optionally normalize landmarks                        │
│     - Shuffle data                                          │
│     - Train with CrossEntropyLoss + Adam                   │
│     - Save to models/posture_model.pth                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Training Parameters (calibrator.train):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 100 | Training iterations over full dataset |
| `batch_size` | 8 | Samples per gradient update |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `use_improved` | False | Use ImprovedPostureClassifier |
| `normalize_landmarks` | True | Apply LandmarkNormalizer |
| `model_name` | 'posture_model.pth' | Output filename |

**Calibration Modes:**

| Mode | What it does |
|------|--------------|
| `fresh` | Discard existing data, collect new |
| `append` | Add new samples to existing data |
| Option 3 | Retrain only (no collection) |
| Option 4 | Retrain with ImprovedPostureClassifier |

---

### 6. `PostureMonitor` (src/monitor.py:21)

**Purpose:** CLI real-time monitoring with OpenCV window

```python
monitor = PostureMonitor(model_path='models/posture_model.pth')
monitor.run()
```

**Features:**
- OpenCV window with pose skeleton overlay
- Green/red border based on prediction
- Prediction smoothing (10-frame buffer)
- Test mode with random predictions (`--test` flag)

---

### 7. `PostureDemo` (app.py:31)

**Purpose:** Gradio-compatible version for web deployment

```python
demo = PostureDemo(model_path='models/posture_model.pth', use_improved=False)
result_frame, trigger_alert = demo.process_frame(frame)
```

**Additional Features over PostureMonitor:**
- Alert system with cooldown
- Returns alert trigger for JavaScript audio
- Frame processing optimized for Gradio streaming

**Alert Logic:**
```python
consecutive_bad = 0      # Counter for bad posture frames
bad_threshold = 3        # Frames before alert
alert_cooldown = 3.0     # Seconds between alerts

if prediction == 0:  # Bad posture
    consecutive_bad += 1
    if consecutive_bad >= bad_threshold and cooldown_passed:
        trigger_alert = True
        consecutive_bad = 0  # Reset
```

---

## Data Flow: End-to-End

### Training Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   WEBCAM     │────►│  MEDIAPIPE   │────►│   STORAGE    │
│              │     │              │     │              │
│ cv2.capture()│     │ 33 landmarks │     │ pickle file  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    MODEL     │◄────│  NORMALIZE   │◄────│    LOAD      │
│              │     │              │     │              │
│ .pth weights │     │ center+scale │     │ good + bad   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Inference Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   WEBCAM     │────►│  MEDIAPIPE   │────►│  NORMALIZE   │
│              │     │              │     │              │
│  RGB frame   │     │ 33 landmarks │     │ 66 floats    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   DISPLAY    │◄────│   SMOOTH     │◄────│    MODEL     │
│              │     │              │     │              │
│ green/red    │     │ 5-frame avg  │     │ softmax prob │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Model Optimization Guide

### Quick Experiments to Try

#### 1. Train with More Data
```bash
cd src
python calibrator.py  # Choose option 2 (append)
# Collect 30 more samples of each
```

#### 2. Train Improved Model
```bash
python calibrator.py  # Choose option 4
# Trains with ResidualBlocks + BatchNorm
```

#### 3. Adjust Training Hyperparameters

Edit `calibrator.py` option 4 section:
```python
calibrator.train(
    X, y,
    num_epochs=300,        # More epochs
    batch_size=16,         # Larger batches (if enough data)
    learning_rate=0.0005,  # Lower LR for fine-tuning
    use_improved=True,
    normalize_landmarks=True,
)
```

#### 4. Modify Model Architecture

Edit `classifier.py` ImprovedPostureClassifier:
```python
# Deeper network
hidden_dims=[256, 128, 64, 32]

# Or wider
hidden_dims=[256, 128]

# More dropout (if overfitting)
dropout=0.5
```

#### 5. Disable Normalization (experiment)

```bash
python calibrator.py --no-normalize
# Choose option 3
```

### Metrics to Watch

During training (`calibrator.py` output):
```
Epoch [  1/100] | Loss: 0.6931 | Accuracy: 50.00%  ◄── Random
Epoch [ 50/100] | Loss: 0.2134 | Accuracy: 92.00%  ◄── Learning
Epoch [100/100] | Loss: 0.0521 | Accuracy: 98.00%  ◄── Converged
```

**Warning Signs:**
- Accuracy stuck at 50% → Model not learning (check data balance)
- Loss not decreasing → Learning rate too high/low
- 100% accuracy too fast → Overfitting (need more data or dropout)

---

## Debugging Tips

### Check if Model is Loading

```python
import torch
from src.classifier import PostureClassifier

model = PostureClassifier()
model.load_state_dict(torch.load('models/posture_model.pth'))
print("Loaded successfully")

# Check weights are not random
print(model.network[0].weight.mean())  # Should be non-zero, small
```

### Inspect Training Data

```python
import pickle

with open('data/training_samples.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Good samples: {len(data['good_samples'])}")
print(f"Bad samples: {len(data['bad_samples'])}")
print(f"Sample shape: {len(data['good_samples'][0])}")  # Should be 66
```

### Test Normalizer

```python
import numpy as np
from src.classifier import LandmarkNormalizer

# Fake landmarks (33 points × 2 coords)
landmarks = np.random.rand(66).astype(np.float32)
normalized = LandmarkNormalizer.normalize(landmarks)

print(f"Input range: [{landmarks.min():.2f}, {landmarks.max():.2f}]")
print(f"Output range: [{normalized.min():.2f}, {normalized.max():.2f}]")
# Output should be roughly centered around 0
```

### Compare Raw vs Normalized Predictions

```python
# In app.py or monitor.py, temporarily add:
print(f"Raw landmarks[0:4]: {landmarks[0:4]}")
print(f"Normalized[0:4]: {normalized[0:4]}")
print(f"Prediction: {pred}, Confidence: {conf:.2%}")
```

---

## Future Improvements

### Model Enhancements
- [ ] Add LSTM/Transformer for temporal modeling
- [ ] Use z-coordinate (depth) from MediaPipe
- [ ] Multi-class posture types (slouch left, slouch right, head forward, etc.)
- [ ] Add angle-based features (spine angle, neck angle)

### Training Enhancements
- [ ] Data augmentation (add noise to landmarks)
- [ ] Cross-validation for hyperparameter tuning
- [ ] Early stopping to prevent overfitting
- [ ] Learning rate scheduling

### Deployment Enhancements
- [ ] Model quantization for faster inference
- [ ] ONNX export for cross-platform deployment
- [ ] TensorFlow Lite for mobile
