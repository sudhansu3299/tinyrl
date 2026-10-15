import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for image-based observations (e.g., CarRacing-v3)."""
    
    def __init__(self, input_channels=3, feature_dim=256):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # Conv block 1: 96x96x3 -> 23x23x32
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Conv block 2: 23x23x32 -> 11x11x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv block 3: 11x11x64 -> 6x6x64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Flatten: 6x6x64 -> 2304
            nn.Flatten(),
            
            # Dense layer: 2304 -> feature_dim
            nn.Linear(2304, feature_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        # Input shape: (batch, height, width, channels) or (batch, channels, height, width)
        # Gym environments typically return (height, width, channels)
        if x.dim() == 4 and x.shape[-1] <= 4:  # (batch, height, width, channels)
            x = x.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
        elif x.dim() == 3 and x.shape[-1] <= 4:  # (height, width, channels)
            x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, channels, height, width)
        
        # Normalize to [0, 1] range if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.cnn(x)
        return features
