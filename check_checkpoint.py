"""Check what epoch the checkpoint is from"""
import torch
import json
from pathlib import Path

# Check if we have training results
if Path('outputs/training_results.json').exists():
    with open('outputs/training_results.json', 'r') as f:
        results = json.load(f)
        print(f"Old training results found:")
        print(f"  Epochs completed: {results['training']['epochs']}")
        print(f"  Images: {results['num_images']}")
        print(f"  Best loss: {results['results']['best_loss']:.4f}")

# Check model file
if Path('outputs/yolo_cct_best.pth').exists():
    checkpoint = torch.load('outputs/yolo_cct_best.pth', map_location='cpu')
    print(f"\nCheckpoint file exists")
    print(f"  File: outputs/yolo_cct_best.pth")
    print(f"  Size: {Path('outputs/yolo_cct_best.pth').stat().st_size / 1024 / 1024:.2f} MB")
    
    # The checkpoint is just state_dict, no epoch info
    # We'll start fresh from epoch 1 using the trained weights
    print(f"\nRecommendation: Start training from epoch 1 with loaded weights")
