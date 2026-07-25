"""
Compact Convolutional Transformer (CCT) Backbone
Replaces traditional Darknet backbone in YOLO with transformer-based feature extraction
"""

import torch
import torch.nn as nn
import math


class ConvTokenizer(nn.Module):
    """Convolutional tokenization - converts image patches to tokens"""
    def __init__(self, in_channels=3, embed_dim=256, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention"""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class CCTBackbone(nn.Module):
    """
    Compact Convolutional Transformer Backbone for YOLO
    Provides better feature extraction through convolutional tokenization and self-attention
    """
    def __init__(self, img_size=416, in_channels=3, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Convolutional tokenization (reduces spatial dimensions)
        self.tokenizer = ConvTokenizer(in_channels, embed_dim)
        
        # Calculate output size after tokenization
        # After 7x7 conv stride 2: 416 -> 208
        # After 3x3 conv stride 2: 208 -> 104
        self.feature_size = img_size // 4
        
        # Positional embedding
        num_patches = self.feature_size * self.feature_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Additional conv layers for YOLO-compatible output
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # Convolutional tokenization: [B, 3, 416, 416] -> [B, 256, 104, 104]
        x = self.tokenizer(x)
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: [B, 256, 104, 104] -> [B, 10816, 256]
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial: [B, 10816, 256] -> [B, 256, 104, 104]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Additional processing for YOLO
        x = self.output_conv(x)
        
        return x


if __name__ == '__main__':
    # Test CCT backbone
    model = CCTBackbone(img_size=416, embed_dim=256, num_layers=4)
    x = torch.randn(1, 3, 416, 416)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
