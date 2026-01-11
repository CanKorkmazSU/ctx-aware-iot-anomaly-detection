import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder 
        # Input to decoder will be 16 x H/4 x W/4
        self.dec1 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, 3, 64, 64]
        x = F.relu(self.enc1(x))
        x = self.pool(x) # [B, 32, 32, 32]
        x = F.relu(self.enc2(x))
        x = self.pool(x) # [B, 16, 16, 16]
        
        x = F.relu(self.dec1(x)) # [B, 32, 32, 32]
        x = torch.sigmoid(self.dec2(x)) # [B, 3, 64, 64]
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MicroAutoencoder(nn.Module):
    """
    Designed for ESP32 constraints (512KB SRAM).
    Input: 32x32x3
    Architecture:
    - Enc: DWSelect(3x3) -> PW(3->8) -> Pool -> DWSelect(3x3) -> PW(8->8)
    """
    def __init__(self):
        super(MicroAutoencoder, self).__init__()
        # Encoder
        # 32x32x3 -> 32x32x8
        self.enc1 = DepthwiseSeparableConv(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 16x16x8 -> 16x16x8
        self.enc2 = DepthwiseSeparableConv(8, 8, kernel_size=3, padding=1)
        
        # Decoder 
        # 8x8x8 input
        # Upsample 8x8 -> 16x16
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        # Upsample 16x16 -> 32x32
        self.dec2 = nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = F.relu(self.enc1(x))
        x = self.pool(x) # [B, 8, 16, 16]
        x = F.relu(self.enc2(x))
        x = self.pool(x) # [B, 8, 8, 8]
        
        x = F.relu(self.dec1(x)) # [B, 8, 16, 16]
        x = torch.sigmoid(self.dec2(x)) # [B, 3, 32, 32]
        return x
