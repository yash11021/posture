"""
PostSURE Flask API
==================
REST API for posture classification, calibration, and training.
Swagger docs available at /apidocs
"""

import os
import sys
import pickle
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import PostureClassifier, LandmarkNormalizer

app = Flask(__name__)
CORS(app)

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger_template = {
    "info": {
        "title": "PostSURE API",
        "description": "REST API for real-time posture classification",
        "version": "1.0.0"
    },
    "basePath": "/",
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Global state
model = None
device = None
model_loaded = False
normalizer = LandmarkNormalizer()

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'posture_model.pth')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_samples.pkl')


def load_model():
    """Load the PyTorch model"""
    global model, device, model_loaded
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PostureClassifier(input_size=66).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            model_loaded = True
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load model: {e}")
            model_loaded = False
    else:
        print(f"No model found at {MODEL_PATH}")
        model_loaded = False


@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: API is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            model_loaded:
              type: boolean
    """
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """
    Get model status
    ---
    responses:
      200:
        description: Model status information
        schema:
          type: object
          properties:
            loaded:
              type: boolean
            device:
              type: string
              example: cpu
            model_path:
              type: string
    """
    return jsonify({
        'loaded': model_loaded,
        'device': str(device) if device else None,
        'model_path': MODEL_PATH
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Classify posture from landmarks
    ---
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - landmarks
          properties:
            landmarks:
              type: array
              items:
                type: number
              description: Array of 66 floats (33 landmarks x 2 coordinates)
              example: [0.5, 0.2, 0.5, 0.2]
    responses:
      200:
        description: Classification result
        schema:
          type: object
          properties:
            prediction:
              type: integer
              description: 0 for bad, 1 for good
            confidence:
              type: number
              description: Confidence score 0-1
            label:
              type: string
              enum: [good, bad]
      400:
        description: Invalid request
      503:
        description: Model not loaded
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'Missing landmarks in request body'}), 400
    
    landmarks = np.array(data['landmarks'], dtype=np.float32)
    
    if len(landmarks) != 66:
        return jsonify({'error': f'Expected 66 landmarks, got {len(landmarks)}'}), 400
    
    try:
        normalized = normalizer.normalize(landmarks)
        
        with torch.no_grad():
            tensor = torch.FloatTensor(normalized).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'label': 'good' if prediction == 1 else 'bad'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """
    Store a labeled sample for training
    ---
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - landmarks
            - label
          properties:
            landmarks:
              type: array
              items:
                type: number
              description: Array of 66 floats
            label:
              type: integer
              enum: [0, 1]
              description: 0 for bad posture, 1 for good
    responses:
      200:
        description: Sample stored successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            total_good:
              type: integer
            total_bad:
              type: integer
      400:
        description: Invalid request
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request body'}), 400
    
    if 'landmarks' not in data or 'label' not in data:
        return jsonify({'error': 'Missing landmarks or label'}), 400
    
    landmarks = data['landmarks']
    label = int(data['label'])
    
    if len(landmarks) != 66:
        return jsonify({'error': f'Expected 66 landmarks, got {len(landmarks)}'}), 400
    
    if label not in [0, 1]:
        return jsonify({'error': 'Label must be 0 (bad) or 1 (good)'}), 400
    
    good_samples = []
    bad_samples = []
    
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            existing = pickle.load(f)
            good_samples = existing.get('good_samples', [])
            bad_samples = existing.get('bad_samples', [])
    
    if label == 1:
        good_samples.append(landmarks)
    else:
        bad_samples.append(landmarks)
    
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'wb') as f:
        pickle.dump({
            'good_samples': good_samples,
            'bad_samples': bad_samples,
            'timestamp': datetime.now().isoformat()
        }, f)
    
    return jsonify({
        'success': True,
        'total_good': len(good_samples),
        'total_bad': len(bad_samples)
    })


@app.route('/api/calibrate/status', methods=['GET'])
def calibrate_status():
    """
    Get calibration data status
    ---
    responses:
      200:
        description: Calibration data status
        schema:
          type: object
          properties:
            has_data:
              type: boolean
            good_samples:
              type: integer
            bad_samples:
              type: integer
            timestamp:
              type: string
    """
    if not os.path.exists(DATA_PATH):
        return jsonify({
            'has_data': False,
            'good_samples': 0,
            'bad_samples': 0
        })
    
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    return jsonify({
        'has_data': True,
        'good_samples': len(data.get('good_samples', [])),
        'bad_samples': len(data.get('bad_samples', [])),
        'timestamp': data.get('timestamp')
    })


@app.route('/api/train', methods=['POST'])
def train():
    """
    Train model with collected samples
    ---
    parameters:
      - in: body
        name: body
        required: false
        schema:
          type: object
          properties:
            epochs:
              type: integer
              default: 100
              description: Number of training epochs
    responses:
      200:
        description: Training completed
        schema:
          type: object
          properties:
            success:
              type: boolean
            epochs:
              type: integer
            accuracy:
              type: number
            samples_used:
              type: integer
      400:
        description: Not enough training data
    """
    global model, model_loaded
    
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No training data available'}), 400
    
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    good_samples = data.get('good_samples', [])
    bad_samples = data.get('bad_samples', [])
    
    if len(good_samples) < 5 or len(bad_samples) < 5:
        return jsonify({
            'error': 'Need at least 5 samples of each class',
            'good_samples': len(good_samples),
            'bad_samples': len(bad_samples)
        }), 400
    
    req_data = request.get_json() or {}
    epochs = req_data.get('epochs', 100)
    
    try:
        X = np.array(good_samples + bad_samples, dtype=np.float32)
        y = np.array([1] * len(good_samples) + [0] * len(bad_samples), dtype=np.int64)
        
        X_normalized = []
        for sample in X:
            try:
                X_normalized.append(normalizer.normalize(sample))
            except:
                X_normalized.append(sample)
        X = np.array(X_normalized, dtype=np.float32)
        
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        train_model = PostureClassifier(input_size=66).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        train_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = train_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        train_model.eval()
        with torch.no_grad():
            outputs = train_model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(train_model.state_dict(), MODEL_PATH)
        
        model = train_model
        model_loaded = True
        
        return jsonify({
            'success': True,
            'epochs': epochs,
            'accuracy': float(accuracy),
            'samples_used': len(X)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Load model on startup
load_model()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
