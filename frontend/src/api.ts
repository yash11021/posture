/**
 * API configuration
 */

// API base URL - uses environment variable or defaults to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:7860';

export interface ClassifyResponse {
    prediction: number;
    confidence: number;
    label: 'good' | 'bad';
}

export interface CalibrateResponse {
    success: boolean;
    total_good: number;
    total_bad: number;
}

export interface CalibrateStatusResponse {
    has_data: boolean;
    good_samples: number;
    bad_samples: number;
    timestamp?: string;
}

export interface TrainResponse {
    success: boolean;
    epochs: number;
    accuracy: number;
    samples_used: number;
}

export interface ModelStatusResponse {
    loaded: boolean;
    device: string | null;
    model_path: string;
}

/**
 * Classify posture from landmarks
 */
export async function classifyPosture(landmarks: number[]): Promise<ClassifyResponse> {
    const response = await fetch(`${API_BASE_URL}/api/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks }),
    });

    if (!response.ok) {
        throw new Error(`Classification failed: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Store a calibration sample
 */
export async function storeSample(landmarks: number[], label: 0 | 1): Promise<CalibrateResponse> {
    const response = await fetch(`${API_BASE_URL}/api/calibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks, label }),
    });

    if (!response.ok) {
        throw new Error(`Calibration failed: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Get calibration data status
 */
export async function getCalibrateStatus(): Promise<CalibrateStatusResponse> {
    const response = await fetch(`${API_BASE_URL}/api/calibrate/status`);
    return response.json();
}

/**
 * Trigger model training
 */
export async function trainModel(epochs: number = 100): Promise<TrainResponse> {
    const response = await fetch(`${API_BASE_URL}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Training failed');
    }

    return response.json();
}

/**
 * Get model status
 */
export async function getModelStatus(): Promise<ModelStatusResponse> {
    const response = await fetch(`${API_BASE_URL}/api/model/status`);
    return response.json();
}
