import torch
import torch.nn as nn
import numpy as np


class PostureClassifier(nn.Module):
    """Simple feedforward network for posture classification (legacy)"""
    def __init__(self, input_size=33*2):  # 33 landmarks x 2 coordinates (x, y)
        super(PostureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Good vs Bad posture
        )

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization for stable training"""
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        # Projection shortcut if dimensions don't match
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out = out + residual  # Skip connection
        return self.relu(out)


class ImprovedPostureClassifier(nn.Module):
    """
    Enhanced posture classifier with:
    - Batch normalization for training stability
    - Residual connections for better gradient flow
    - Configurable architecture
    - Support for normalized landmark features
    """
    def __init__(self, input_size=66, num_classes=2, hidden_dims=[128, 64, 32], dropout=0.3):
        super(ImprovedPostureClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size)

        # Build residual blocks
        layers = []
        in_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)
        # Extract features through residual blocks
        features = self.feature_extractor(x)
        # Classify
        return self.classifier(features)

    def get_confidence(self, x):
        """Get prediction with confidence score"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            return prediction, confidence, probs


class LandmarkNormalizer:
    """
    Normalize MediaPipe landmarks to be position and scale invariant.

    This makes the model robust to:
    - Different positions in the frame (sitting left vs right)
    - Different distances from camera (close vs far)
    - Different body sizes
    """

    # Key landmark indices from MediaPipe Pose
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0

    @staticmethod
    def normalize(landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize 66-element landmark array (33 landmarks x 2 coords).

        Args:
            landmarks: Raw [x0, y0, x1, y1, ..., x32, y32] from MediaPipe

        Returns:
            Normalized landmarks centered at hip midpoint, scaled by shoulder width
        """
        if len(landmarks) != 66:
            raise ValueError(f"Expected 66 landmarks, got {len(landmarks)}")

        # Reshape to (33, 2) for easier manipulation
        points = landmarks.reshape(33, 2)

        # Calculate hip center (origin point)
        left_hip = points[LandmarkNormalizer.LEFT_HIP]
        right_hip = points[LandmarkNormalizer.RIGHT_HIP]
        hip_center = (left_hip + right_hip) / 2

        # Calculate shoulder width for scale normalization
        left_shoulder = points[LandmarkNormalizer.LEFT_SHOULDER]
        right_shoulder = points[LandmarkNormalizer.RIGHT_SHOULDER]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        # Avoid division by zero
        if shoulder_width < 0.01:
            shoulder_width = 0.01

        # Center and scale
        normalized = (points - hip_center) / shoulder_width

        return normalized.flatten().astype(np.float32)

    @staticmethod
    def compute_posture_angles(landmarks: np.ndarray) -> np.ndarray:
        """
        Compute meaningful angles that indicate posture quality.

        Returns angles for:
        - Head tilt (nose relative to shoulder midpoint)
        - Shoulder alignment (left vs right shoulder height)
        - Spine angle (shoulder center to hip center)
        """
        points = landmarks.reshape(33, 2)

        # Get key points
        nose = points[LandmarkNormalizer.NOSE]
        left_shoulder = points[LandmarkNormalizer.LEFT_SHOULDER]
        right_shoulder = points[LandmarkNormalizer.RIGHT_SHOULDER]
        left_hip = points[LandmarkNormalizer.LEFT_HIP]
        right_hip = points[LandmarkNormalizer.RIGHT_HIP]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # Head forward tilt angle (how far nose is forward from shoulder line)
        head_forward = nose[1] - shoulder_center[1]  # y difference

        # Shoulder level difference (should be ~0 for good posture)
        shoulder_tilt = left_shoulder[1] - right_shoulder[1]

        # Spine angle (vertical alignment from hips to shoulders)
        spine_vector = shoulder_center - hip_center
        spine_angle = np.arctan2(spine_vector[0], spine_vector[1])  # Angle from vertical

        # Neck angle (nose to shoulder center vs vertical)
        neck_vector = nose - shoulder_center
        neck_angle = np.arctan2(neck_vector[0], neck_vector[1])

        return np.array([head_forward, shoulder_tilt, spine_angle, neck_angle], dtype=np.float32)