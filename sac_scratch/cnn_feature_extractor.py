import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for image based inputs like Carracing-v3"""

    def __init__(self, input_channels=4, feature_dim=256):
        """CNN feature extractor for stacked grayscale frames.

        Default input_channels=4 matches 4 stacked grayscale frames (CarRacing-v3).
        """
        super().__init__()

        self.cnn = nn.Sequential(
            # Conv block 1: 96x96x4 -> 23x23x32 (H_out = (H_in + 2*padding - kernel_size)/Stride + 1)
            # per patch of kernel, so, 4x8x8(1 filter shape) -> 1 value; 32 such filters
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(), # Layer 1: broader initial patterns
            
            # Conv block 2: 23x23x32 -> 11x11x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), #padding is chosen to preserve edge info, control shrinking aggressively
            nn.ReLU(), #Layer 2: Extract patterns for smaller spatial scale
            
            # Conv block 3: 11x11x64 -> 6x6(spatial size of each map)x64(feature maps/channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), #Layer 3: Extract patterns for even smaller spatial scale
            
            # Flatten: 6x6x64 -> 2304
            nn.Flatten(),
            
            # Dense layer: 2304 -> feature_dim
            # this layer combines entire spatial info into a compact feature representation
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
        
        # Normalize to [0, 1] range as nn train in rightly scaled inputs
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.cnn(x) # 256 features dim vector
        return features
