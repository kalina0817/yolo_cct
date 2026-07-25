"""
YOLO Detection Head
Converts backbone features to object detection predictions
"""

import torch
import torch.nn as nn


class YOLOHead(nn.Module):
    """
    YOLO detection head for object detection
    Outputs: [batch, anchors * (5 + num_classes), grid_h, grid_w]
    Each anchor predicts: (x, y, w, h, objectness, class_probs...)
    """
    def __init__(self, in_channels=256, num_classes=20, num_anchors=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each anchor predicts: x, y, w, h, objectness + class probabilities
        self.predictions_per_anchor = 5 + num_classes
        out_channels = num_anchors * self.predictions_per_anchor
        
        # Detection convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Final prediction layer
        self.pred_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        predictions = self.pred_conv(x)
        return predictions


if __name__ == '__main__':
    # Test YOLO head
    head = YOLOHead(in_channels=256, num_classes=20, num_anchors=3)
    x = torch.randn(2, 256, 104, 104)
    out = head(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: [batch, {3 * (5 + 20)}, 104, 104]")
    print(f"Parameters: {sum(p.numel() for p in head.parameters()):,}")
