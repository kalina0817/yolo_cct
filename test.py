"""
YOLO-CCT Test Script
Run inference on sample images
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import random

import sys
sys.path.append(str(Path(__file__).parent))

from models import YOLO_CCT
from utils import VOC_CLASSES


def load_model(model_path='outputs/yolo_cct_best.pth', device='cpu'):
    """Load trained model"""
    model = YOLO_CCT(
        img_size=224,
        num_classes=20,
        num_anchors=3,
        embed_dim=128,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def preprocess_image(image_path, img_size=224):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size
    
    # Resize
    img_resized = img.resize((img_size, img_size))
    
    # Convert to tensor
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img, img_tensor, orig_size


def decode_predictions(predictions, conf_threshold=0.01, grid_size=56):
    """Decode YOLO predictions to bounding boxes"""
    num_anchors = 3
    num_classes = 20
    
    # Reshape
    pred = predictions.view(1, num_anchors, 5 + num_classes, grid_size, grid_size)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    
    detections = []
    
    for a in range(num_anchors):
        for i in range(grid_size):
            for j in range(grid_size):
                # Get prediction
                pred_xy = torch.sigmoid(pred[0, a, i, j, 0:2])
                pred_wh = pred[0, a, i, j, 2:4]
                pred_conf = torch.sigmoid(pred[0, a, i, j, 4])
                pred_cls = F.softmax(pred[0, a, i, j, 5:], dim=0)
                
                if pred_conf > conf_threshold:
                    # Convert to absolute coordinates
                    x = (j + pred_xy[0].item()) / grid_size
                    y = (i + pred_xy[1].item()) / grid_size
                    w = pred_wh[0].item()
                    h = pred_wh[1].item()
                    
                    class_id = torch.argmax(pred_cls).item()
                    class_conf = pred_cls[class_id].item()
                    
                    confidence = pred_conf.item() * class_conf
                    
                    detections.append({
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': VOC_CLASSES[class_id],
                        'bbox': [x, y, w, h]
                    })
    
    return detections


def draw_detections(image, detections, orig_size):
    """Draw bounding boxes on image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    width, height = orig_size
    
    # Colors for different classes
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8'
    ] * 2
    
    for det in detections:
        x, y, w, h = det['bbox']
        
        # Convert normalized coordinates to pixel coordinates
        x_center = x * width
        y_center = y * height
        box_w = w * width
        box_h = h * height
        
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        # Draw box
        color = colors[det['class_id']]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        
        # Draw label background
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill='white', font=font)
    
    return img_draw


def test_images(model_path='outputs/yolo_cct_best.pth', num_images=5):
    """Test model on random images"""
    
    print("\n" + "="*80)
    print(" " * 30 + "YOLO-CCT TESTING")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load model
    print(f"\n📦 Loading Model...")
    model = load_model(model_path, device)
    print(f"   ✓ Model loaded from: {model_path}")
    
    # Find test images
    print(f"\n🖼️  Finding Test Images...")
    dataset_path = Path('../voc2012_dataset')
    
    # Try different paths
    possible_paths = [
        dataset_path / 'VOC2012_test/VOC2012_test/JPEGImages',
        dataset_path / 'VOC2012_train_val/VOC2012_train_val/JPEGImages'
    ]
    
    image_files = []
    for path in possible_paths:
        if path.exists():
            image_files = list(path.glob('*.jpg'))
            if image_files:
                break
    
    if not image_files:
        print("   ✗ No test images found in archive")
        return
    
    # Select random images
    test_images = random.sample(image_files, min(num_images, len(image_files)))
    print(f"   ✓ Found {len(image_files)} images, testing {len(test_images)}")
    
    # Create output directory
    output_dir = Path('outputs/test_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test each image
    print(f"\n🔍 Running Inference...")
    
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n   [{idx}/{len(test_images)}] {img_path.name}")
        
        # Load and preprocess
        img, img_tensor, orig_size = preprocess_image(img_path)
        img_tensor = img_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Decode predictions (lowered threshold to detect more)
        detections = decode_predictions(predictions, conf_threshold=0.01)
        
        # Debug: Check max confidence
        pred_reshaped = predictions.view(1, 3, 25, 56, 56).permute(0, 1, 3, 4, 2)
        max_conf = torch.sigmoid(pred_reshaped[0, :, :, :, 4]).max().item()
        print(f"       Max confidence: {max_conf:.4f}")
        print(f"       Detections: {len(detections)}")
        
        # Draw and save
        if detections:
            img_with_boxes = draw_detections(img, detections, orig_size)
            output_path = output_dir / f'test_{idx}_{img_path.stem}.jpg'
            img_with_boxes.save(output_path)
            
            print(f"       Objects:")
            for det in detections[:5]:  # Show top 5
                print(f"         - {det['class_name']}: {det['confidence']:.3f}")
            
            print(f"       ✓ Saved: {output_path}")
        else:
            print(f"       No objects detected above threshold")
    
    # Summary
    print(f"\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   - Tested {len(test_images)} images")
    print(f"   - Results saved to: {output_dir}/")
    print(f"\n💡 Tips:")
    print(f"   - Adjust confidence threshold in decode_predictions() for more/fewer detections")
    print(f"   - View results in outputs/test_results/")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    test_images(num_images=5)
