"""
YOLO-CCT Evaluation Script
Calculates mAP, FPS, and generates comparison report
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))

from models import YOLO_CCT
from utils import VOCDataset, VOC_CLASSES


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def decode_predictions(predictions, conf_threshold=0.5, grid_size=104):
    """
    Decode YOLO predictions to bounding boxes
    Returns: list of (confidence, class_id, [x, y, w, h]) for each detection
    """
    batch_size = predictions.shape[0]
    num_anchors = 3
    num_classes = 20
    
    # Reshape: [B, A*(5+C), H, W] -> [B, A, H, W, 5+C]
    pred = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    
    detections = []
    
    for b in range(batch_size):
        batch_detections = []
        
        for a in range(num_anchors):
            for i in range(grid_size):
                for j in range(grid_size):
                    # Get prediction
                    pred_xy = torch.sigmoid(pred[b, a, i, j, 0:2])
                    pred_wh = pred[b, a, i, j, 2:4]
                    pred_conf = torch.sigmoid(pred[b, a, i, j, 4])
                    pred_cls = F.softmax(pred[b, a, i, j, 5:], dim=0)
                    
                    if pred_conf > conf_threshold:
                        # Convert to absolute coordinates
                        x = (j + pred_xy[0].item()) / grid_size
                        y = (i + pred_xy[1].item()) / grid_size
                        w = pred_wh[0].item()
                        h = pred_wh[1].item()
                        
                        class_id = torch.argmax(pred_cls).item()
                        class_conf = pred_cls[class_id].item()
                        
                        confidence = pred_conf.item() * class_conf
                        
                        batch_detections.append((confidence, class_id, [x, y, w, h]))
        
        detections.append(batch_detections)
    
    return detections


def calculate_map(model, dataloader, device, conf_threshold=0.5, iou_threshold=0.5):
    """Calculate mean Average Precision (mAP)"""
    model.eval()
    
    # Store predictions and ground truths for each class
    class_predictions = {i: [] for i in range(20)}
    class_ground_truths = {i: [] for i in range(20)}
    
    print("Calculating mAP...")
    
    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(dataloader):
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            images = images.to(device)
            predictions = model(images)
            
            # Decode predictions
            detections = decode_predictions(predictions, conf_threshold, grid_size=56)
            
            # Process each image in batch
            for img_idx in range(len(images)):
                # Get ground truth boxes
                for a in range(3):
                    for i in range(56):
                        for j in range(56):
                            if targets[img_idx, a, i, j, 4] > 0.5:  # Has object
                                x = (j + targets[img_idx, a, i, j, 0].item()) / 56
                                y = (i + targets[img_idx, a, i, j, 1].item()) / 56
                                w = targets[img_idx, a, i, j, 2].item()
                                h = targets[img_idx, a, i, j, 3].item()
                                class_id = torch.argmax(targets[img_idx, a, i, j, 5:]).item()
                                
                                class_ground_truths[class_id].append([x, y, w, h])
                
                # Store predictions
                for conf, class_id, box in detections[img_idx]:
                    class_predictions[class_id].append((conf, box))
    
    # Calculate AP for each class
    aps = []
    
    for class_id in range(20):
        preds = class_predictions[class_id]
        gts = class_ground_truths[class_id]
        
        if len(gts) == 0:
            continue
        
        if len(preds) == 0:
            aps.append(0.0)
            continue
        
        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x[0], reverse=True)
        
        # Match predictions to ground truths
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        gt_matched = [False] * len(gts)
        
        for pred_idx, (conf, pred_box) in enumerate(preds):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP (area under precision-recall curve)
        ap = 0
        for i in range(len(precisions) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
        
        aps.append(ap)
    
    mAP = np.mean(aps) if aps else 0.0
    return mAP, aps


def measure_fps(model, device, img_size=416, num_iterations=100):
    """Measure inference speed (FPS)"""
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_iterations / total_time
    
    return fps


def evaluate_model(model_path='outputs/yolo_cct_best.pth'):
    """Main evaluation function"""
    
    print("\n" + "="*80)
    print(" " * 25 + "YOLO-CCT EVALUATION")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load model
    print(f"\n📦 Loading Model...")
    model = YOLO_CCT(
        img_size=224,
        num_classes=20,
        num_anchors=3,
        embed_dim=128,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model_info = model.get_model_info()
    
    print(f"   ✓ Loaded from: {model_path}")
    print(f"   ✓ Parameters: {model_info['total_parameters']:,}")
    
    # Load dataset
    print(f"\n📂 Loading Dataset...")
    data_path = Path('../voc2012_dataset/VOC2012_train_val/VOC2012_train_val')
    dataset = VOCDataset(data_path, split='trainval', img_size=224, grid_size=56)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Calculate mAP
    print(f"\n📊 Calculating mAP...")
    mAP, class_aps = calculate_map(model, dataloader, device, conf_threshold=0.01)
    print(f"   ✓ mAP@0.5: {mAP:.4f} ({mAP*100:.2f}%)")
    
    # Measure FPS
    print(f"\n⚡ Measuring FPS...")
    fps = measure_fps(model, device, img_size=224)
    print(f"   ✓ FPS: {fps:.2f}")
    
    # Generate comparison report
    print(f"\n" + "="*80)
    print(" " * 25 + "COMPARISON REPORT")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Baseline YOLO':<20} {'YOLO-CCT':<20}")
    print("-" * 65)
    params_str = f'{model_info["total_parameters"]/1e6:.2f}M'
    size_str = f'{model_info["model_size_mb"]:.2f} MB'
    map_str = f'{mAP*100:.2f}%'
    fps_str = f'{fps:.2f}'
    print(f"{'Parameters':<25} {'11.5M':<20} {params_str:<20}")
    print(f"{'Model Size':<25} {'~46 MB':<20} {size_str:<20}")
    print(f"{'mAP@0.5':<25} {'~57.9%':<20} {map_str:<20}")
    print(f"{'FPS (CPU)':<25} {'~10-15':<20} {fps_str:<20}")
    print(f"{'Backbone':<25} {'Darknet-19':<20} {'CCT (Transformer)':<20}")
    
    # Save results
    results = {
        'model': 'YOLO-CCT',
        'evaluation': {
            'mAP': mAP,
            'mAP_percentage': mAP * 100,
            'fps': fps,
            'class_aps': {VOC_CLASSES[i]: ap for i, ap in enumerate(class_aps) if i < len(class_aps)}
        },
        'model_info': model_info,
        'comparison': {
            'baseline_yolo': {
                'parameters': '11.5M',
                'model_size': '~46 MB',
                'mAP': '~57.9%',
                'fps': '~10-15'
            },
            'yolo_cct': {
                'parameters': f'{model_info["total_parameters"]/1e6:.2f}M',
                'model_size': f'{model_info["model_size_mb"]:.2f} MB',
                'mAP': f'{mAP*100:.2f}%',
                'fps': f'{fps:.2f}'
            }
        }
    }
    
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved evaluation results: outputs/evaluation_results.json")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    evaluate_model()
