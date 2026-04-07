"""
U-Net 1D architecture for audio super-resolution.

Architecture improvements:
- Dilated convolutions at bottleneck for expanded receptive field
- Interpolation-based upsampling to prevent checkerboard artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x) + self.residual(x))


class DilatedBottleneck(nn.Module):
    """Bottleneck with dilated convolutions for expanded receptive field."""

    def __init__(self, channels, kernel_size=7, dilation_rates=[1, 2, 4, 8, 16]):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilation_rates:
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class AudioUNet1D(nn.Module):
    """
    U-Net 1D for audio super-resolution with improved architecture.
    
    Improvements:
    - Residual connections in encoder/decoder blocks
    - Dilated bottleneck for expanded receptive field
    - Interpolation-based upsampling to prevent checkerboard artifacts
    - Larger base channels for better feature extraction
    """

    def __init__(self, in_channels=1, base_channels=48, depth=4, use_dilated_bottleneck=True, use_interpolation_upsampling=True):
        super().__init__()

        self.depth = depth
        self.use_interpolation_upsampling = use_interpolation_upsampling
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()

        channels = in_channels
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.encoder_convs.append(DoubleConv(channels, out_channels))
            self.encoder_pools.append(nn.MaxPool1d(2))
            channels = out_channels

        if use_dilated_bottleneck:
            self.bottleneck = DilatedBottleneck(channels, kernel_size=7, dilation_rates=[1, 2, 4, 8, 16])
            self.bottleneck_proj = nn.Conv1d(channels, channels * 2, 1)
        else:
            self.bottleneck = DoubleConv(channels, channels * 2)
            self.bottleneck_proj = None

        channels = channels * 2
        for i in range(depth):
            skip_channels = base_channels * (2 ** (depth - 1 - i))
            out_channels = skip_channels // 2 if i < depth - 1 else base_channels

            if use_interpolation_upsampling:
                self.decoder_ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.decoder_ups.append(nn.Conv1d(channels, channels // 2, kernel_size=7, padding=3))
            else:
                self.decoder_ups.append(nn.ConvTranspose1d(channels, channels // 2, kernel_size=2, stride=2))
            self.decoder_convs.append(DoubleConv(channels // 2 + skip_channels, out_channels))
            channels = out_channels

        self.output_conv = nn.Conv1d(channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        input_signal = x
        skips = []

        for i in range(self.depth):
            x = self.encoder_convs[i](x)
            skips.append(x)
            x = self.encoder_pools[i](x)

        x = self.bottleneck(x)
        if self.bottleneck_proj is not None:
            x = self.bottleneck_proj(x)

        for i in range(self.depth):
            if self.use_interpolation_upsampling:
                x = self.decoder_ups[i * 2](x)
                x = self.decoder_ups[i * 2 + 1](x)
            else:
                x = self.decoder_ups[i](x)
            skip = skips[-(i + 1)]

            if x.shape[-1] < skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            elif x.shape[-1] > skip.shape[-1]:
                x = x[:, :, :skip.shape[-1]]

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_convs[i](x)

        x = self.output_conv(x)
        x = x + input_signal
        return x.squeeze(1)


class AudioUNet1DSimple(nn.Module):
    """
    Very simple U-Net for quick testing.
    """

    def __init__(self, in_channels=1, channels=32, use_interpolation_upsampling=True):
        super().__init__()
        self.use_interpolation_upsampling = use_interpolation_upsampling

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, channels, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 4, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.bottleneck = DilatedBottleneck(channels * 4, kernel_size=7, dilation_rates=[1, 2, 4, 8, 16])

        if use_interpolation_upsampling:
            self.dec3_up = nn.Upsample(scale_factor=2, mode='nearest')
            self.dec3_conv = nn.Conv1d(channels * 4, channels * 2, kernel_size=7, padding=3)
            self.dec2_up = nn.Upsample(scale_factor=2, mode='nearest')
            self.dec2_conv = nn.Conv1d(channels * 4, channels, kernel_size=7, padding=3)
        else:
            self.dec3_conv = nn.ConvTranspose1d(channels * 4, channels * 2, 4, stride=2, padding=1)
            self.dec2_conv = nn.ConvTranspose1d(channels * 4, channels, 4, stride=2, padding=1)

        self.dec1 = nn.Conv1d(channels * 2, in_channels, 7, padding=3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        if self.use_interpolation_upsampling:
            d3 = self.dec3_up(b)
            d3 = self.dec3_conv(d3)
        else:
            d3 = self.dec3_conv(b)

        if d3.shape[-1] < e2.shape[-1]:
            d3 = F.pad(d3, (0, e2.shape[-1] - d3.shape[-1]))
        d3 = torch.cat([d3, e2], dim=1)

        if self.use_interpolation_upsampling:
            d2 = self.dec2_up(d3)
            d2 = self.dec2_conv(d2)
        else:
            d2 = self.dec2_conv(d3)

        if d2.shape[-1] < e1.shape[-1]:
            d2 = F.pad(d2, (0, e1.shape[-1] - d2.shape[-1]))
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)

        return d1.squeeze(1)


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AudioUNet1DSimple(in_channels=1, channels=32)
    print(f"AudioUNet1DSimple parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 65536)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
