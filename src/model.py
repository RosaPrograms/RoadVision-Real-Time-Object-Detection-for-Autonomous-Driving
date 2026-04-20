import torch
import torch.nn as nn

# Conv → BatchNorm → LeakyReLU : Follows normal darknet pattern

class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # 'same' padding for odd kernels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Darknet-style residual block: 1x1 conv (bottleneck) → 3x3 conv → skip connection
# The bottleneck halves channels then restores them, reducing parameters while preserving spatial features.
class DarknetResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.block = nn.Sequential(
            ConvBNLeaky(channels, mid, kernel_size=1),   # pointwise
            ConvBNLeaky(mid, channels, kernel_size=3),   # depthwise spatial
        )

    def forward(self, x):
        return x + self.block(x)  # residual / skip connection

# Each stage consists of a downsampling conv followed by N residual blocks.
class DarknetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        # stride=2 halves spatial resolution (like a max-pool but learnable)
        self.downsample = ConvBNLeaky(in_channels, out_channels,
                                      kernel_size=3, stride=2)
        self.blocks = nn.Sequential(
            *[DarknetResidualBlock(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.downsample(x))
    
# Full Backbone: Initial stem + 4 stages with increasing channels and decreasing spatial resolution.
class RoadVisionDarknetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial stem
        # 3 -> 32 channels, 512x512 -> 256x256
        self.stem = ConvBNLeaky(3, 32, kernel_size=3, stride=2)

        # Stage 1: 256x256 -> 128x128, 32 -> 64 channels, 1 residual block
        self.stage1 = DarknetStage(32, 64, num_blocks=1)

        # Stage 2: 128x128 -> 64x64, 64 -> 128 channels, 2 residual blocks
        self.stage2 = DarknetStage(64, 128, num_blocks=2)

        # Stage 3: 64x64 -> 32x32, 128 -> 256 channels, 8 residual blocks
        self.stage3 = DarknetStage(128, 256, num_blocks=8)

        # Stage 4: 32x32 -> 16x16, 256 -> 512 channels, 8 residual blocks
        self.stage4 = DarknetStage(256, 512, num_blocks=8)

        # weight initialization
        self._initialize_weights()

    # Kaiming He initialization for conv layers, BatchNorm weights to 1 and bias to 0
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)      # (B, 32, 256, 256)
        x = self.stage1(x)    # (B, 64, 128, 128)
        p3 = self.stage2(x)   # (B, 128, 64, 64)
        p4 = self.stage3(p3)  # (B, 256, 32, 32)
        p5 = self.stage4(p4)  # (B, 512, 16, 16)
        return p3, p4, p5
    
# check
if __name__ == "__main__":
    backbone = RoadVisionDarknetBackbone()

    # count trainable parameters
    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total trainable parameters in RoadVisionDarknetBackbone: {total_params:,}")

    # verify output shapes with a dummy input
    dummy = torch.zeros(2, 3, 512, 512)  # batch of 2 images
    p3, p4, p5 = backbone(dummy)
    print(f"P3 shape: {p3.shape}")  # Expected: (2, 128, 64, 64)
    print(f"P4 shape: {p4.shape}")  # Expected: (2, 256, 32, 32)
    print(f"P5 shape: {p5.shape}")  # Expected: (2, 512, 16, 16)