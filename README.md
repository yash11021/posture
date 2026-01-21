---
title: PostSURE
emoji: 🧘
colorFrom: green
colorTo: blue
sdk: docker
app_file: api/app.py
pinned: false
license: mit
---

# PostSURE - Real-time Posture Classification

A microservices-based posture classification system using MediaPipe (browser) and PyTorch (server).

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Frontend       │   API   │  Backend        │
│  (Vercel)       │◄───────►│  (HF Spaces)    │
│                 │         │                 │
│  TypeScript     │         │  Flask API      │
│  MediaPipe JS   │         │  PyTorch Model  │
│  Vite           │         │  Gunicorn       │
└─────────────────┘         └─────────────────┘
```

## Quick Links

- **Frontend**: [[postsure.vercel.app](https://posture-aa4qzsu53-yash11021s-projects.vercel.app)]
- **API Docs**: [postSURE.hf.space/apidocs](https://yashrajsinha-postsure.hf.space/apidocs/)

## Features

- **Real-time pose detection** in browser (MediaPipe JS - no server round-trip for video)
- **REST API** for classification, calibration, and training
- **Swagger documentation** at `/apidocs/`
- **Calibration mode** to train personalized model

## Project Structure

```
posture/
├── api/
│   └── app.py           # Flask REST API with Swagger
├── frontend/
│   ├── src/
│   │   ├── main.ts      # MediaPipe + UI
│   │   ├── api.ts       # API client
│   │   └── style.css    # Dark theme
│   ├── index.html
│   └── vercel.json
├── src/
│   └── classifier.py    # PyTorch model classes
├── models/
│   └── posture_model.pth
├── requirements.txt
└── Dockerfile
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/classify` | POST | Classify 66 landmarks |
| `/api/calibrate` | POST | Store training sample |
| `/api/calibrate/status` | GET | Get sample counts |
| `/api/train` | POST | Retrain model |
| `/api/model/status` | GET | Model info |

## Development

### Backend
```bash
pip install -r requirements.txt
python api/app.py
# API at http://localhost:8080
# Swagger at http://localhost:8080/apidocs/
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

### Environment Variables

**Frontend** (`.env.local` or Vercel):
```
VITE_API_URL=http://localhost:8080
```

## Deployment

### Backend → HuggingFace Spaces
Automatic via GitHub Actions on push to `main` (backend files only).

### Frontend → Vercel
1. Link GitHub repo to Vercel
2. Set root directory: `frontend`
3. Set env var: `VITE_API_URL=https://yashrajsinha-postsure.hf.space`

## License

MIT
