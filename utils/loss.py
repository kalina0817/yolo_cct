"""
YOLO Loss Function
Combines objectness, bounding box, and classification losses
"""

import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    YOLO Detection Loss
    Combines:
    - Localization loss (bounding box coordinates)
    - Objectness loss (confidence that object exists)
    - Classification loss (which class)
    """
    def __init__(self, num_classes=20, num_anchors=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Loss functions
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        
        # Loss weights (following YOLO paper)
        self.lambda_coord = 5.0  # Coordinate loss weight
        self.lambda_noobj = 0.5  # No-object confidence weight
        self.lambda_obj = 1.0    # Object confidence weight
        self.lambda_cls = 1.0    # Classification loss weight
        
    def forward(self, predictions, targets, num_boxes):
        """
        Args:
            predictions: [batch, anchors*(5+classes), grid_h, grid_w]
            targets: [batch, anchors, grid_h, grid_w, 5+classes]
            num_boxes: total number of objects in batch
            
        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual loss components
        """
        batch_size = predictions.shape[0]
        grid_size = predictions.shape[-1]
        
        # Reshape predictions to match target format
        # [B, A*(5+C), H, W] -> [B, A, H, W, 5+C]
        pred = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract prediction components
        pred_xy = torch.sigmoid(pred[..., 0:2])      # x, y (apply sigmoid)
        pred_wh = pred[..., 2:4]                     # w, h (raw)
        pred_conf_logits = pred[..., 4:5]            # objectness logits
        pred_cls_logits = pred[..., 5:]              # class logits
        
        # Extract target components
        target_xy = targets[..., 0:2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4:5]
        target_cls = targets[..., 5:]
        
        # Object mask (cells that contain objects)
        obj_mask = target_conf > 0.5
        noobj_mask = ~obj_mask
        
        # Expand mask for coordinate dimensions (remove last dim for xy/wh)
        obj_mask_coord = obj_mask.squeeze(-1)
        
        # 1. Coordinate Loss (only for cells with objects)
        if obj_mask_coord.sum() > 0:
            loss_xy = self.mse(pred_xy[obj_mask_coord], target_xy[obj_mask_coord])
            loss_wh = self.mse(pred_wh[obj_mask_coord], target_wh[obj_mask_coord])
        else:
            loss_xy = torch.tensor(0.0, device=predictions.device)
            loss_wh = torch.tensor(0.0, device=predictions.device)
        
        # 2. Objectness Loss
        # For cells with objects
        if obj_mask.sum() > 0:
            loss_conf_obj = self.bce(pred_conf_logits[obj_mask], target_conf[obj_mask])
        else:
            loss_conf_obj = torch.tensor(0.0, device=predictions.device)
        
        # For cells without objects
        if noobj_mask.sum() > 0:
            loss_conf_noobj = self.bce(pred_conf_logits[noobj_mask], target_conf[noobj_mask])
        else:
            loss_conf_noobj = torch.tensor(0.0, device=predictions.device)
        
        # 3. Classification Loss (only for cells with objects)
        if obj_mask.sum() > 0:
            obj_mask_flat = obj_mask.squeeze(-1)
            pred_cls_obj = pred_cls_logits[obj_mask_flat]
            target_cls_obj = target_cls[obj_mask_flat]
            target_cls_idx = target_cls_obj.argmax(dim=-1)
            loss_cls = self.ce(pred_cls_obj, target_cls_idx)
        else:
            loss_cls = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses with weights
        total_loss = (
            self.lambda_coord * loss_xy +
            self.lambda_coord * loss_wh +
            self.lambda_obj * loss_conf_obj +
            self.lambda_noobj * loss_conf_noobj +
            self.lambda_cls * loss_cls
        )
        
        # Normalize by batch size and number of boxes
        if num_boxes > 0:
            total_loss = total_loss / (batch_size * max(num_boxes, 1))
        
        # Return loss and components for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'xy': loss_xy.item(),
            'wh': loss_wh.item(),
            'conf_obj': loss_conf_obj.item(),
            'conf_noobj': loss_conf_noobj.item(),
            'cls': loss_cls.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test loss function
    criterion = YOLOLoss(num_classes=20, num_anchors=3)
    
    batch_size = 2
    grid_size = 104
    
    # Dummy predictions and targets
    predictions = torch.randn(batch_size, 3 * (5 + 20), grid_size, grid_size)
    targets = torch.zeros(batch_size, 3, grid_size, grid_size, 5 + 20)
    
    # Add a fake object in first batch
    targets[0, 0, 50, 50, 0:2] = torch.tensor([0.5, 0.5])  # xy
    targets[0, 0, 50, 50, 2:4] = torch.tensor([0.2, 0.3])  # wh
    targets[0, 0, 50, 50, 4] = 1.0  # objectness
    targets[0, 0, 50, 50, 5] = 1.0  # class 0
    
    num_boxes = 1
    
    loss, loss_dict = criterion(predictions, targets, num_boxes)
    
    print("YOLO Loss Test:")
    print(f"Total Loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value:.4f}")
