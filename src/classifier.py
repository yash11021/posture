import torch
import torch.nn as nn


class PostureClassifier(nn.Module):
    """Simple feedforward network for posture classification"""
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