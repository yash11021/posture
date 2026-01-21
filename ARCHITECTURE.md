# PostSURE Architecture

## System Overview

PostSURE is a real-time posture classification system split into two deployable units:

```
┌─────────────────────────────────────────────────────────────┐
│                         USER BROWSER                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    FRONTEND (Vercel)                   │ │
│  │                                                        │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │ │
│  │  │   Webcam    │───►│ MediaPipe JS│───►│  66 floats │ │ │
│  │  │             │    │ Pose Detect │    │ (landmarks)│ │ │
│  │  └─────────────┘    └─────────────┘    └─────┬──────┘ │ │
│  │                                              │        │ │
│  │  ┌─────────────┐    ┌─────────────┐          │        │ │
│  │  │     UI      │◄───│  API Client │◄─────────┘        │ │
│  │  │ (skeleton,  │    │ (TypeScript)│                   │ │
│  │  │  status)    │    └──────┬──────┘                   │ │
│  │  └─────────────┘           │                          │ │
│  └────────────────────────────┼───────────────────────────┘ │
│                               │ HTTPS                        │
└───────────────────────────────┼──────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    BACKEND (HuggingFace Spaces)                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                      Flask API                            │ │
│  │                                                           │ │
│  │  /api/classify ────► LandmarkNormalizer ────► PyTorch    │ │
│  │  /api/calibrate ───► pickle storage                      │ │
│  │  /api/train ───────► training loop ────► save model      │ │
│  │  /apidocs ─────────► Swagger UI                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐                     │
│  │ posture_model   │  │ training_samples│                     │
│  │ .pth            │  │ .pkl            │                     │
│  └─────────────────┘  └─────────────────┘                     │
└───────────────────────────────────────────────────────────────┘
```

## Data Flow

### Real-time Classification
1. Browser captures webcam frame
2. MediaPipe JS extracts 33 pose landmarks (x, y coords = 66 floats)
3. Frontend sends landmarks to `/api/classify`
4. Backend normalizes → classifies → returns `{prediction, confidence, label}`
5. Frontend updates UI (border color, status text)

### Calibration
1. User captures pose in "good" or "bad" position
2. Frontend sends landmarks + label to `/api/calibrate`
3. Backend appends to `training_samples.pkl`

### Training
1. User triggers training via `/api/train`
2. Backend loads samples, normalizes, trains PyTorch model
3. Saves new `posture_model.pth`

## Model Architecture

```
Input (66 features: 33 landmarks × 2 coordinates)
    │
    ▼
┌─────────────────────────────────┐
│ Linear(66 → 128) + ReLU + Drop  │
├─────────────────────────────────┤
│ Linear(128 → 64) + ReLU + Drop  │
├─────────────────────────────────┤
│ Linear(64 → 32) + ReLU          │
├─────────────────────────────────┤
│ Linear(32 → 2)                  │
└─────────────────────────────────┘
    │
    ▼
Output: [P(bad), P(good)]
```

## Landmark Normalization

Raw MediaPipe coords are position/scale dependent. `LandmarkNormalizer`:

1. **Centers** at hip midpoint (translation invariance)
2. **Scales** by shoulder width (scale invariance)

## File Structure

```
posture/
├── api/
│   └── app.py              # Flask API + Swagger
├── frontend/
│   ├── src/
│   │   ├── main.ts         # MediaPipe + UI logic
│   │   ├── api.ts          # Typed API client
│   │   └── style.css       # Dark theme
│   ├── index.html
│   ├── package.json
│   └── vercel.json
├── src/
│   ├── classifier.py       # PostureClassifier, LandmarkNormalizer
│   ├── calibrator.py       # CLI training tool (optional)
│   └── monitor.py          # CLI monitor (optional)
├── models/
│   └── posture_model.pth
├── data/
│   └── training_samples.pkl
├── requirements.txt
├── Dockerfile
└── .github/workflows/
    └── sync.yml            # Deploy backend to HF
```

## Deployment

| Component | Platform | Trigger |
|-----------|----------|---------|
| Backend | HuggingFace Spaces | Push to `main` (api/, src/, models/) |
| Frontend | Vercel | Push to `main` (frontend/) |
