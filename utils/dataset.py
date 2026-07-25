"""
Pascal VOC Dataset Loader
Loads images and annotations from Pascal VOC format
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    """
    Pascal VOC dataset loader
    Supports training and validation splits
    """
    def __init__(self, root_dir, split='trainval', img_size=416, grid_size=104, num_anchors=3):
        """
        Args:
            root_dir: Path to VOC root directory (e.g., VOC2012_train_val/VOC2012_train_val)
            split: 'train', 'val', or 'trainval'
            img_size: Target image size (default: 416)
            grid_size: YOLO grid size (default: 104 for img_size 416)
            num_anchors: Number of anchors per grid cell (default: 3)
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = len(VOC_CLASSES)
        
        # Load image IDs from ImageSets
        imageset_path = self.root_dir / f'ImageSets/Main/{split}.txt'
        if not imageset_path.exists():
            # Fallback to trainval if specific split not found
            imageset_path = self.root_dir / 'ImageSets/Main/trainval.txt'
            if not imageset_path.exists():
                imageset_path = self.root_dir / 'ImageSets/Main/train.txt'
        
        with open(imageset_path) as f:
            self.image_ids = [line.strip() for line in f]
        
        self.img_dir = self.root_dir / 'JPEGImages'
        self.ann_dir = self.root_dir / 'Annotations'
        
        print(f"Loaded {len(self.image_ids)} images from {imageset_path}")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = self.img_dir / f'{img_id}.jpg'
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Load annotations
        ann_path = self.ann_dir / f'{img_id}.xml'
        boxes, classes = self._parse_annotation(ann_path, orig_w, orig_h)
        
        # Resize image
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Create target tensor for YOLO loss
        target = self._create_target(boxes, classes)
        
        return img_tensor, target, len(boxes)
    
    def _parse_annotation(self, xml_path, img_w, img_h):
        """Parse VOC XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        classes = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Skip if class not in VOC_CLASSES
            if class_name not in CLASS_TO_IDX:
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Normalize coordinates to [0, 1]
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h
            
            # Clamp to valid range
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            boxes.append([x_center, y_center, width, height])
            classes.append(CLASS_TO_IDX[class_name])
        
        return boxes, classes
    
    def _create_target(self, boxes, classes):
        """
        Create target tensor for YOLO loss
        Returns: [num_anchors, grid_h, grid_w, 5 + num_classes]
        """
        target = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 5 + self.num_classes))
        
        for box, cls in zip(boxes, classes):
            x_center, y_center, width, height = box
            
            # Convert normalized coordinates to grid coordinates
            grid_x = x_center * self.grid_size
            grid_y = y_center * self.grid_size
            
            grid_i = int(grid_x)
            grid_j = int(grid_y)
            
            # Check bounds
            if not (0 <= grid_i < self.grid_size and 0 <= grid_j < self.grid_size):
                continue
            
            # Assign to first anchor (simplified - in full YOLO, use IoU with anchor boxes)
            anchor_idx = 0
            
            # Cell-relative coordinates (offset within grid cell)
            x_offset = grid_x - grid_i
            y_offset = grid_y - grid_j
            
            # Fill target
            target[anchor_idx, grid_j, grid_i, 0] = x_offset
            target[anchor_idx, grid_j, grid_i, 1] = y_offset
            target[anchor_idx, grid_j, grid_i, 2] = width
            target[anchor_idx, grid_j, grid_i, 3] = height
            target[anchor_idx, grid_j, grid_i, 4] = 1.0  # objectness
            target[anchor_idx, grid_j, grid_i, 5 + cls] = 1.0  # class one-hot
        
        return target


if __name__ == '__main__':
    # Test dataset
    dataset = VOCDataset(
        root_dir='../archive/VOC2012_train_val/VOC2012_train_val',
        split='trainval',
        img_size=416,
        grid_size=104
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load a sample
    img, target, num_boxes = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Number of objects: {num_boxes}")
