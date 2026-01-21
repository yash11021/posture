import './style.css';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import { classifyPosture, storeSample, getCalibrateStatus, trainModel, getModelStatus } from './api';

// DOM Elements
const video = document.getElementById('webcam') as HTMLVideoElement;
const canvas = document.getElementById('overlay') as HTMLCanvasElement;
const videoContainer = document.getElementById('video-container') as HTMLDivElement;
const statusText = document.getElementById('status-text') as HTMLSpanElement;
const confidenceText = document.getElementById('confidence') as HTMLSpanElement;
const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const modeMonitor = document.getElementById('mode-monitor') as HTMLButtonElement;
const modeCalibrate = document.getElementById('mode-calibrate') as HTMLButtonElement;
const calibrationPanel = document.getElementById('calibration-panel') as HTMLDivElement;
const captureGoodBtn = document.getElementById('capture-good') as HTMLButtonElement;
const captureBadBtn = document.getElementById('capture-bad') as HTMLButtonElement;
const goodCountSpan = document.getElementById('good-count') as HTMLSpanElement;
const badCountSpan = document.getElementById('bad-count') as HTMLSpanElement;
const trainBtn = document.getElementById('train-btn') as HTMLButtonElement;

// State
let poseLandmarker: PoseLandmarker | null = null;
let isRunning = false;
let currentMode: 'monitor' | 'calibrate' = 'monitor';
let latestLandmarks: number[] | null = null;
let goodCount = 0;
let badCount = 0;

// Drawing context
const ctx = canvas.getContext('2d')!;
let drawingUtils: DrawingUtils;

/**
 * Initialize MediaPipe Pose Landmarker
 */
async function initializePoseLandmarker(): Promise<void> {
  statusText.textContent = 'Loading model...';

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numPoses: 1
  });

  drawingUtils = new DrawingUtils(ctx);
  statusText.textContent = 'Ready - Click Start';
}

/**
 * Start webcam stream
 */
async function startCamera(): Promise<void> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: 640, height: 480 }
    });

    video.srcObject = stream;
    await video.play();

    // Match canvas size to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    isRunning = true;
    startBtn.textContent = 'Stop Camera';
    statusText.textContent = 'Detecting...';

    // Start detection loop
    detectPose();
  } catch (err) {
    console.error('Camera error:', err);
    statusText.textContent = 'Camera access denied';
  }
}

/**
 * Stop webcam stream
 */
function stopCamera(): void {
  const stream = video.srcObject as MediaStream;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  video.srcObject = null;
  isRunning = false;
  startBtn.textContent = 'Start Camera';
  statusText.textContent = 'Camera stopped';
  videoContainer.className = '';
}

/**
 * Main pose detection loop
 */
async function detectPose(): Promise<void> {
  if (!isRunning || !poseLandmarker) return;

  const startTime = performance.now();
  const result = poseLandmarker.detectForVideo(video, startTime);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (result.landmarks && result.landmarks.length > 0) {
    const landmarks = result.landmarks[0];

    // Draw skeleton
    drawingUtils.drawLandmarks(landmarks, {
      radius: 3,
      color: '#00FF00',
      fillColor: '#00FF00'
    });

    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 2
    });

    // Extract landmarks as flat array (66 values: x, y for 33 landmarks)
    latestLandmarks = landmarks.flatMap(l => [l.x, l.y]);

    // In monitor mode, classify posture
    if (currentMode === 'monitor') {
      try {
        const result = await classifyPosture(latestLandmarks);

        // Update UI
        videoContainer.className = result.label;
        statusText.textContent = result.label === 'good' ? 'Good Posture ✓' : 'Bad Posture - Sit up!';
        confidenceText.textContent = `${Math.round(result.confidence * 100)}%`;
      } catch (err) {
        // Model might not be loaded
        statusText.textContent = 'Pose detected (no model)';
        videoContainer.className = '';
        confidenceText.textContent = '';
      }
    } else {
      statusText.textContent = 'Ready to capture';
      videoContainer.className = '';
      confidenceText.textContent = '';
    }
  } else {
    latestLandmarks = null;
    statusText.textContent = 'No pose detected';
    videoContainer.className = '';
    confidenceText.textContent = '';
  }

  // Continue loop
  requestAnimationFrame(detectPose);
}

/**
 * Capture a calibration sample
 */
async function captureSample(label: 0 | 1): Promise<void> {
  if (!latestLandmarks) {
    alert('No pose detected - make sure you are visible in the camera');
    return;
  }

  try {
    const result = await storeSample(latestLandmarks, label);
    goodCount = result.total_good;
    badCount = result.total_bad;
    updateCounts();

    // Flash feedback
    videoContainer.className = label === 1 ? 'good' : 'bad';
    setTimeout(() => { videoContainer.className = ''; }, 200);
  } catch (err) {
    console.error('Failed to store sample:', err);
    alert('Failed to store sample. Is the API running?');
  }
}

/**
 * Update sample count display
 */
function updateCounts(): void {
  goodCountSpan.textContent = goodCount.toString();
  badCountSpan.textContent = badCount.toString();
  trainBtn.disabled = goodCount < 5 || badCount < 5;
}

/**
 * Switch between monitor and calibrate modes
 */
function setMode(mode: 'monitor' | 'calibrate'): void {
  currentMode = mode;

  modeMonitor.className = mode === 'monitor' ? 'active' : '';
  modeCalibrate.className = mode === 'calibrate' ? 'active' : '';
  calibrationPanel.style.display = mode === 'calibrate' ? 'block' : 'none';

  if (mode === 'calibrate') {
    // Load current counts
    getCalibrateStatus().then(status => {
      goodCount = status.good_samples;
      badCount = status.bad_samples;
      updateCounts();
    }).catch(() => {
      // API might not be running
    });
  }
}

/**
 * Train the model
 */
async function handleTrain(): Promise<void> {
  trainBtn.disabled = true;
  trainBtn.textContent = 'Training...';

  try {
    const result = await trainModel(100);
    alert(`Training complete!\nAccuracy: ${Math.round(result.accuracy * 100)}%\nSamples used: ${result.samples_used}`);
    trainBtn.textContent = 'Train Model';
    trainBtn.disabled = false;
  } catch (err) {
    alert(`Training failed: ${err}`);
    trainBtn.textContent = 'Train Model';
    trainBtn.disabled = false;
  }
}

// Event listeners
startBtn.addEventListener('click', () => {
  if (isRunning) {
    stopCamera();
  } else {
    startCamera();
  }
});

modeMonitor.addEventListener('click', () => setMode('monitor'));
modeCalibrate.addEventListener('click', () => setMode('calibrate'));
captureGoodBtn.addEventListener('click', () => captureSample(1));
captureBadBtn.addEventListener('click', () => captureSample(0));
trainBtn.addEventListener('click', handleTrain);

// Initialize
initializePoseLandmarker().then(() => {
  // Check model status
  getModelStatus().then(status => {
    if (!status.loaded) {
      console.log('Model not loaded - calibration may be needed');
    }
  }).catch(() => {
    console.log('API not available - running in frontend-only mode');
  });
});
