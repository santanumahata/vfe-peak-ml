"""Fully convolutional 1D U-Net that emits one heatmap channel per peak (P0..P4)."""

import torch                                                     # - | tensors and autograd
import torch.nn as nn                                            # - | layers


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
