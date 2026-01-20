---
title: PostSURE
emoji: 🧘
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# Real-time Posture Classification

A PyTorch-based posture classification system using MediaPipe pose estimation. Detects good vs bad sitting posture in real-time from webcam input.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/YOUR_USERNAME/posture-classifier)** *(update link after deployment)*

## Features

- **Real-time pose detection** using MediaPipe (33 body landmarks)
- **Custom PyTorch neural network** with residual connections and batch normalization
- **Landmark normalization** for position/scale invariant predictions
- **Temporal smoothing** to reduce prediction jitter
- **Web interface** via Gradio for easy demo access
- **Training pipeline** with data collection and model training

## Architecture

### Model Comparison

| Feature | Basic Model | Improved Model |
|---------|-------------|----------------|
| Architecture | 3-layer MLP | Residual blocks |
| Normalization | None | BatchNorm + Landmark normalization |
| Parameters | ~19K | ~25K |
| Skip connections | No | Yes |

### Improved Model Architecture

```
Input (66 features: 33 landmarks × 2 coordinates)
    ↓
BatchNorm1d(66)
    ↓
┌─────────────────────────────────────┐
│ ResidualBlock(66 → 128)             │
│   Linear → BatchNorm → ReLU → Dropout│
│   Linear → BatchNorm                 │
│   + skip connection                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ ResidualBlock(128 → 64)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ ResidualBlock(64 → 32)              │
└─────────────────────────────────────┘
    ↓
Linear(32 → 16) → ReLU → Dropout
    ↓
Linear(16 → 2) → Softmax
    ↓
Output: [P(bad), P(good)]
```

### Landmark Normalization

Raw MediaPipe coordinates are position and scale dependent. The `LandmarkNormalizer` transforms them to be invariant:

1. **Center** landmarks at hip midpoint (translation invariance)
2. **Scale** by shoulder width (scale invariance)

This allows the model to generalize across different:
- Positions in the frame (sitting left vs right)
- Distances from camera (close vs far)
- Body sizes

## Project Structure

```
posture/
├── app.py                  # Gradio web interface (HuggingFace Spaces)
├── src/
│   ├── classifier.py       # PyTorch model definitions
│   ├── calibrator.py       # Data collection and training
│   ├── monitor.py          # Real-time inference (OpenCV)
│   └── setup.sh            # Environment setup
├── models/
│   └── posture_model.pth   # Trained model weights
├── data/
│   └── training_samples.pkl # Collected training data
├── requirements.txt        # Local development dependencies
└── requirements-hf.txt     # HuggingFace Spaces dependencies
```

## Quick Start

### Prerequisites

- Python 3.11.9 (required for MediaPipe compatibility)
- Webcam

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Demo

```bash
python app.py
```

Open http://localhost:7860 in your browser.

### Calibrate Your Own Model

1. **Collect training data:**
   ```bash
   cd src
   python calibrator.py
   ```
   Choose option 1 (fresh start), then follow prompts to capture good and bad posture samples.

2. **Train with improved architecture:**
   ```bash
   python calibrator.py
   ```
   Choose option 4 to retrain with the improved model.

### Run CLI Monitor

```bash
python src/monitor.py         # With trained model
python src/monitor.py --test  # Demo mode (random predictions)
```

## Deployment to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Clone and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/posture-classifier
   git push hf main
   ```

The Space will automatically use `app.py` and `requirements-hf.txt`.

## Technical Details

### MediaPipe Pose Landmarks

33 keypoints are extracted per frame:

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | Nose | 11-12 | Shoulders |
| 1-4 | Eyes & Ears | 13-14 | Elbows |
| 5-6 | Mouth | 15-16 | Wrists |
| 7-8 | Inner Eyes | 23-24 | Hips |
| 9-10 | Outer Eyes | 25-28 | Knees & Ankles |

### Training Pipeline

1. **Data Collection**: Capture labeled pose samples via webcam
2. **Preprocessing**: Normalize landmarks (center + scale)
3. **Training**: PyTorch with Adam optimizer, CrossEntropyLoss
4. **Validation**: Per-epoch accuracy tracking
5. **Export**: Save state dict to `.pth` file

### Inference Pipeline

1. **Capture**: Webcam frame via OpenCV
2. **Detect**: MediaPipe extracts 33 landmarks
3. **Normalize**: Center at hips, scale by shoulders
4. **Classify**: PyTorch forward pass
5. **Smooth**: Rolling average over 5-10 frames
6. **Display**: Visual feedback (green/red border)

## Future Improvements

- [ ] Multi-class posture types (slouching, leaning, etc.)
- [ ] Temporal modeling with LSTM/Transformer
- [ ] Use z-coordinate (depth) from MediaPipe
- [ ] Angle-based features (spine angle, head tilt)
- [ ] Mobile deployment with TensorFlow Lite

## License

MIT

---

Built with PyTorch, MediaPipe, and Gradio.
