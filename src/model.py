"""Fully convolutional 1D U-Net that emits one heatmap channel per peak (P0..P4)."""

import numpy as np                                               # - | input scaling
import torch                                                     # - | tensors and autograd
import torch.nn as nn                                            # - | layers


def normalise(X):                                                # - | per-trace scaling
    """Zero-median, unit-max scaling so absolute amps never matter."""
    med = np.median(X, axis=-1, keepdims=True)                   # O(Nn) | robust baseline per trace
    out = X - med                                                # O(Nn) | remove the pedestal
    scale = np.max(np.abs(out), axis=-1, keepdims=True) + 1e-6   # O(Nn) | peak-to-peak scale
    return (out / scale).astype(np.float32)                      # O(Nn) | scaled trace
    # Total: time O(N n), space O(N n)


def conv_block(c_in, c_out):                                     # - | two convs with norm and activation
    """Standard U-Net double convolution, kernel 9 to span a spiky peak's width."""
    return nn.Sequential(                                        # - | one sequential unit
        nn.Conv1d(c_in, c_out, 9, padding=4),                    # O(n*c_in*c_out) | widen receptive field
        nn.BatchNorm1d(c_out),                                   # O(n*c_out) | stabilise training
        nn.ReLU(inplace=True),                                   # O(n*c_out) | nonlinearity
        nn.Conv1d(c_out, c_out, 9, padding=4),                   # O(n*c_out^2) | second conv
        nn.BatchNorm1d(c_out),                                   # O(n*c_out) | normalise again
        nn.ReLU(inplace=True),                                   # O(n*c_out) | nonlinearity
    )
    # Total: time O(n * c_in * c_out), space O(n * c_out)


class PeakUNet(nn.Module):
    """Three-level U-Net; input (B,1,n) trace, output (B,5,n) peak logits."""

    def __init__(self, base=16, n_peaks=5):                      # - | width and number of output channels
        super().__init__()                                       # - | init nn.Module
        self.enc1 = conv_block(1, base)                          # - | level 1 encoder
        self.enc2 = conv_block(base, base * 2)                   # - | level 2 encoder
        self.enc3 = conv_block(base * 2, base * 4)               # - | level 3 encoder
        self.bottom = conv_block(base * 4, base * 8)             # - | bottleneck, widest receptive field
        self.pool = nn.MaxPool1d(2)                              # - | halve the time axis
        self.up3 = nn.ConvTranspose1d(base * 8, base * 4, 2, 2)  # - | upsample back to level 3
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 2, 2)  # - | upsample back to level 2
        self.up1 = nn.ConvTranspose1d(base * 2, base, 2, 2)      # - | upsample back to level 1
        self.dec3 = conv_block(base * 8, base * 4)               # - | decoder after skip concat
        self.dec2 = conv_block(base * 4, base * 2)               # - | decoder after skip concat
        self.dec1 = conv_block(base * 2, base)                   # - | decoder after skip concat
        self.head = nn.Conv1d(base, n_peaks, 1)                  # - | per-sample logit per peak channel

    def forward(self, x):                                        # - | one forward pass
        e1 = self.enc1(x)                                        # O(n) | level 1 features
        e2 = self.enc2(self.pool(e1))                            # O(n/2) | level 2 features
        e3 = self.enc3(self.pool(e2))                            # O(n/4) | level 3 features
        b = self.bottom(self.pool(e3))                           # O(n/8) | global context
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))          # O(n/4) | merge context with level 3
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))         # O(n/2) | merge with level 2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))         # O(n) | merge with level 1
        return self.head(d1)                                     # O(n) | five heatmap logits
        # Total: time O(n * base^2), space O(n * base)


def peaks_from_logits(logits):                                   # - | decode heatmaps into positions
    """argmax per channel plus its sigmoid confidence."""
    prob = torch.sigmoid(logits)                                 # O(5n) | logits to [0,1]
    conf, pos = prob.max(dim=-1)                                 # O(5n) | best sample per peak channel
    return pos, conf                                             # - | (B,5) positions and confidences
    # Total: time O(B * 5n), space O(B * 5n)


class PeakUNetN(nn.Module):
    """U-Net with a configurable number of levels. Channels double per level up to
    `cap`. levels=3 has the same layout as PeakUNet; each extra level halves the
    time axis once more, doubling how much of the trace the bottleneck sees."""

    def __init__(self, base=32, levels=4, n_peaks=5, cap=512):  # - | width, depth, outputs
        super().__init__()                                       # - | init nn.Module
        ch = [min(base * 2 ** i, cap) for i in range(levels + 1)]  # - | width per level, last = bottleneck
        self.enc = nn.ModuleList([conv_block(1 if i == 0 else ch[i - 1], ch[i])
                                  for i in range(levels)])       # - | encoders
        self.bottom = conv_block(ch[levels - 1], ch[levels])     # - | bottleneck
        self.up = nn.ModuleList([nn.ConvTranspose1d(ch[i + 1], ch[i], 2, 2)
                                 for i in range(levels)])        # - | upsamplers
        self.dec = nn.ModuleList([conv_block(2 * ch[i], ch[i]) for i in range(levels)])  # - | decoders
        self.pool = nn.MaxPool1d(2)                              # - | halve the time axis
        self.head = nn.Conv1d(ch[0], n_peaks, 1)                 # - | per-sample logits

    def forward(self, x):                                        # - | (B,1,n) -> (B,5,n); n divisible by 2^levels
        skips = []                                               # - | encoder outputs
        for i, e in enumerate(self.enc):                         # O(levels) | down path
            x = e(x if i == 0 else self.pool(x))                 # O(n/2^i) | features
            skips.append(x)                                      # - | keep for the skip
        x = self.bottom(self.pool(x))                            # O(n/2^L) | widest context
        for i in reversed(range(len(self.enc))):                 # O(levels) | up path
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))  # O(n/2^i) | merge
        return self.head(x)                                      # O(n) | logits
        # Total: time O(n * base^2 * levels), space O(n * base)
