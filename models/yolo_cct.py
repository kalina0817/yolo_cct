"""
YOLO-CCT: Complete model combining CCT backbone with YOLO detection head
"""

import torch
import torch.nn as nn
from .cct_backbone import CCTBackbone
from .yolo_head import YOLOHead


class YOLO_CCT(nn.Module):
    """
    YOLO with Compact Convolutional Transformer Backbone
    
    Architecture:
    - CCT Backbone: Transformer-based feature extraction (replaces Darknet)
    - YOLO Head: Object detection with bounding boxes and class predictions
    
    Benefits:
    - Better feature extraction through self-attention
    - More efficient than standard YOLO (fewer parameters)
    - Better at detecting small and complex objects
    """
    def __init__(self, img_size=416, num_classes=20, num_anchors=3, 
                 embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # CCT Backbone (transformer-based)
        self.backbone = CCTBackbone(
            img_size=img_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # YOLO Detection Head
        self.head = YOLOHead(
            in_channels=embed_dim,
            num_classes=num_classes,
            num_anchors=num_anchors
        )
        
        # Calculate grid size
        self.grid_size = img_size // 4  # After tokenization stride
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input images [batch, 3, img_size, img_size]
        Returns:
            predictions: [batch, anchors*(5+classes), grid_h, grid_w]
        """
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions
    
    def get_model_info(self):
        """Get model information for evaluation"""
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        return {
            'total_parameters': total_params,
            'backbone_parameters': backbone_params,
            'head_parameters': head_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # FP32
            'grid_size': self.grid_size,
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors
        }


if __name__ == '__main__':
    # Test full model
    model = YOLO_CCT(img_size=416, num_classes=20, num_anchors=3, 
                     embed_dim=256, num_layers=4, num_heads=8)
    
    x = torch.randn(2, 3, 416, 416)
    out = model(x)
    
    print("="*60)
    print("YOLO-CCT Model Test")
    print("="*60)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    print("="*60)
