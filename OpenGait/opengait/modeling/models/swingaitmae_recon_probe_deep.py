import torch.nn as nn

from .swingaitmae_recon_probe import SwinGaitmaeFrozenEncoderReconProbe


class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualConv3DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ReconstructionDecoder3D(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNReLU3D(in_channels, 256),
            ResidualConv3DBlock(256),
            ConvBNReLU3D(256, 128),
            ResidualConv3DBlock(128),
            ConvBNReLU3D(128, 64),
            ConvBNReLU3D(64, 32),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)


class SwinGaitmaeFrozenEncoderDeepReconProbe(SwinGaitmaeFrozenEncoderReconProbe):
    """Freeze encoder and ID heads, and train deep reconstruction decoders only."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.decoder = ReconstructionDecoder3D(in_channels=512, out_channels=1)
        self.decoder_edge = ReconstructionDecoder3D(in_channels=512, out_channels=1)