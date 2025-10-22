
# Posture Corrector

Lightweight posture-correction project using MediaPipe, OpenCV and PyTorch.

This repository contains tools to calibrate a user's neutral posture, run real-time classification of posture from webcam frames, and monitor posture over time. A pre-trained PyTorch model is included in `models/posture_model.pth`.

## Features

- Calibrate a neutral posture baseline (`src/calibrator.py`).
- Run posture classification on live webcam input (`src/classifier.py`).
- Monitor posture continuously and optionally emit events or logs (`src/monitor.py`).
- Uses MediaPipe for pose landmarks, OpenCV for image capture/visualization, and a PyTorch model for classification.

## Repo layout

- `src/` - source scripts
	- `calibrator.py` - Collect baseline pose data and save calibration file.
	- `classifier.py` - Load the model and classify posture from webcam frames.
	- `monitor.py` - Higher-level monitoring loop, integrates classifier and optional alerting.
	- `setup.sh` - Convenience script to create a virtualenv and install dependencies.
- `models/` - trained model artifacts
	- `posture_model.pth` - pre-trained PyTorch model used by `classifier.py`.
- `data/` - (optional) place to store captured calibration or training data.
- `requirements.txt` - Python dependencies used by the project.

## Quickstart

Prerequisites: Python 3.11.9 (strictly required).

Important: MediaPipe currently publishes prebuilt wheels that are compatible with specific Python versions. At the time this project was created, MediaPipe support for newer Python versions (for example 3.12+) is not yet available on all platforms. For this reason this project targets Python 3.11.9 specifically — using a different Python minor/patch version may cause installation or runtime failures.

To make this easier, `src/setup.sh` automates installing and pinning Python 3.11.9 using `pyenv`, creating a virtual environment, and installing the pinned dependencies. If you already have a working Python 3.11.x environment you can still follow the quickstart steps.

1. Create and activate a virtual environment (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Use the provided helper to set up the environment:

```bash
bash src/setup.sh
```

4. Calibrate your neutral posture (recommended before monitoring):

```bash
python src/calibrator.py
```

Follow on-screen instructions. Calibration will create a small JSON or npy file in `data/` (if implemented) used by the classifier.

5. Run the classifier on webcam input:

```bash
python src/classifier.py --model models/posture_model.pth
```

6. Start the monitor to track posture and log or notify over time:

```bash
python src/monitor.py
```

## Arguments and configuration

- `--model` - path to the PyTorch model file (default: `models/posture_model.pth`).
- `--camera` - camera device index or path (default: 0).
- `--threshold` - classification threshold or sensitivity for alerts.

Check individual scripts for their full argument lists and configurable options. They typically use argparse; run `python src/<script>.py -h` for details.

## Implementation notes

- Landmark extraction is done with MediaPipe Pose. The scripts read pose landmarks, perform optional normalization against the calibration baseline, and pass a fixed-size feature vector into the PyTorch model.
- The model expects a numeric input (numpy/tensor) of shape consistent with `models/posture_model.pth` training. If you need to re-train, export the training pipeline and matching normalization steps.

## Troubleshooting

- Camera not found: ensure no other application is using the webcam and try a different device index (`--camera 1`).
- Model load errors: confirm the model file exists and was saved with a compatible PyTorch version. If the repo's model was trained with a different torch version, create a matching environment or re-export the model.
- MediaPipe errors: ensure your platform supports the installed MediaPipe wheel. On macOS, the pip wheel in `requirements.txt` should work for most setups.