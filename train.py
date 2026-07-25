"""
YOLO-CCT Training Script
Trains the model on Pascal VOC dataset with proper YOLO loss
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from models import YOLO_CCT
from utils import VOCDataset, YOLOLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0.0
    epoch_components = {
        'xy': 0.0,
        'wh': 0.0,
        'conf_obj': 0.0,
        'conf_noobj': 0.0,
        'cls': 0.0
    }
    
    num_batches = len(dataloader)
    
    for batch_idx, (images, targets, num_boxes_list) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        num_boxes = sum(num_boxes_list).item()
        
        # Forward pass
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets, num_boxes)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_loss += loss.item()
        for key in epoch_components:
            epoch_components[key] += loss_dict[key]
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"  [{epoch}/{total_epochs}] Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
    
    # Calculate averages
    avg_loss = epoch_loss / num_batches
    avg_components = {k: v / num_batches for k, v in epoch_components.items()}
    
    return avg_loss, avg_components


def train_model(epochs=25, batch_size=2, learning_rate=0.001, img_size=224, resume=False):
    """Main training function"""
    
    print("\n" + "="*80)
    print(" " * 20 + "YOLO-CCT TRAINING")
    print("  Compact Convolutional Transformer Backbone + YOLO Detection")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print(f"\n📂 Loading Pascal VOC Dataset...")
    data_path = Path('../voc2012_dataset/VOC2012_train_val/VOC2012_train_val')
    
    if not data_path.exists():
        print(f"   Error: Dataset not found at {data_path}")
        print("   Please ensure archive folder contains VOC2012_train_val/VOC2012_train_val")
        return
    
    grid_size = img_size // 4  # CCT reduces by factor of 4
    
    # Use subset of dataset to reduce memory usage
    full_dataset = VOCDataset(
        root_dir=data_path,
        split='trainval',
        img_size=img_size,
        grid_size=grid_size,
        num_anchors=3
    )
    
    # Full training with 2000 images
    from torch.utils.data import Subset
    dataset = Subset(full_dataset, range(min(2000, len(full_dataset))))
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"   ✓ Loaded {len(dataset)} images")
    print(f"   ✓ Batches per epoch: {len(dataloader)}")
    
    # Create model
    print(f"\n🏗️  Building YOLO-CCT Model...")
    model = YOLO_CCT(
        img_size=img_size,
        num_classes=20,
        num_anchors=3,
        embed_dim=128,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    model_info = model.get_model_info()
    print(f"   ✓ Total Parameters: {model_info['total_parameters']:,}")
    print(f"   ✓ Model Size: {model_info['model_size_mb']:.2f} MB")
    print(f"   ✓ Grid Size: {model_info['grid_size']}x{model_info['grid_size']}")
    
    # Loss and optimizer
    criterion = YOLOLoss(num_classes=20, num_anchors=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if requested
    start_epoch = 1
    best_loss = float('inf')
    history = {
        'epochs': [],
        'losses': [],
        'loss_components': []
    }
    previous_epochs = 0
    
    if resume and Path('outputs/yolo_cct_best.pth').exists():
        print(f"\n🔄 Loading pretrained weights from checkpoint...")
        checkpoint = torch.load('outputs/yolo_cct_best.pth', map_location=device)
        model.load_state_dict(checkpoint)
        
        # Check previous training info
        if Path('outputs/training_results.json').exists():
            with open('outputs/training_results.json', 'r') as f:
                results = json.load(f)
                best_loss = results['results'].get('best_loss', float('inf'))
                previous_epochs = results['training'].get('epochs', 0)
        
        print(f"   ✓ Loaded trained weights (previous best loss: {best_loss:.4f})")
        print(f"   ✓ Previous training: {previous_epochs} epochs completed")
        print(f"   ✓ Continuing training for {epochs} more epochs")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {start_epoch} to {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Image Size: {img_size}x{img_size}")
    
    # Training loop
    print("\n" + "="*80)
    print("🚀 Starting Training...")
    print("="*80)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        # Train one epoch
        avg_loss, avg_components = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, epochs
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Total Loss: {avg_loss:.4f}")
        print(f"   Components:")
        print(f"     - XY Loss:       {avg_components['xy']:.4f}")
        print(f"     - WH Loss:       {avg_components['wh']:.4f}")
        print(f"     - Obj Loss:      {avg_components['conf_obj']:.4f}")
        print(f"     - NoObj Loss:    {avg_components['conf_noobj']:.4f}")
        print(f"     - Class Loss:    {avg_components['cls']:.4f}")
        print(f"   Time: {epoch_time/60:.1f} minutes")
        
        # Save history
        history['epochs'].append(epoch)
        history['losses'].append(avg_loss)
        history['loss_components'].append(avg_components)
        
        # Save checkpoint every epoch to prevent data loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }, f'outputs/checkpoint_epoch_{epoch}.pth')
        print(f"   ✓ Saved checkpoint: epoch_{epoch}.pth")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'outputs/yolo_cct_best.pth')
            print(f"   ✓ Saved best model (loss: {best_loss:.4f})")
    
    total_time = time.time() - start_time
    
    # Save final model and results
    print("\n" + "="*80)
    print("💾 Saving Results...")
    print("="*80)
    
    torch.save(model.state_dict(), 'outputs/yolo_cct_final.pth')
    
    results = {
        'model': 'YOLO-CCT',
        'architecture': {
            'backbone': 'Compact Convolutional Transformer (CCT)',
            'head': 'YOLO Detection Head',
            'embed_dim': 128,
            'num_layers': 2,
            'num_heads': 4
        },
        'dataset': 'Pascal VOC 2012',
        'num_images': len(dataset),
        'num_classes': 20,
        'training': {
            'epochs': previous_epochs + epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'img_size': img_size,
            'total_time_minutes': total_time / 60,
            'time_per_epoch_minutes': total_time / 60 / epochs,
            'resumed_from_epoch': previous_epochs
        },
        'model_info': model_info,
        'results': {
            'final_loss': history['losses'][-1],
            'best_loss': best_loss,
            'history': history
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('outputs/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Saved final model: outputs/yolo_cct_final.pth")
    print(f"   ✓ Saved best model: outputs/yolo_cct_best.pth")
    print(f"   ✓ Saved results: outputs/training_results.json")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📈 Final Results:")
    print(f"   Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Final Loss: {history['losses'][-1]:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model Parameters: {model_info['total_parameters']:,}")
    print(f"   Model Size: {model_info['model_size_mb']:.2f} MB")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run evaluation: python evaluate.py")
    print(f"   2. Test on images: python test.py")
    print(f"   3. Compare with baseline YOLO")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Full training: 2000 images, 25 epochs (~11 hours)
    # Saves checkpoint every epoch to prevent data loss
    train_model(
        epochs=25,
        batch_size=2,
        learning_rate=0.001,
        img_size=224,
        resume=False
    )
