"""
Quick status check for training progress
"""
import json
from pathlib import Path

outputs_dir = Path('outputs')

print("\n" + "="*60)
print("YOLO-CCT Training Status")
print("="*60)

# Check if training results exist
results_file = outputs_dir / 'training_results.json'
best_model = outputs_dir / 'yolo_cct_best.pth'
final_model = outputs_dir / 'yolo_cct_final.pth'

if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n✅ Training Complete!")
    print(f"\nFinal Results:")
    print(f"  Final Loss: {results['results']['final_loss']:.4f}")
    print(f"  Best Loss: {results['results']['best_loss']:.4f}")
    print(f"  Epochs: {results['training']['epochs']}")
    print(f"  Training Time: {results['training']['total_time_minutes']:.1f} minutes")
    print(f"\nModel Info:")
    print(f"  Parameters: {results['model_info']['total_parameters']:,}")
    print(f"  Size: {results['model_info']['model_size_mb']:.2f} MB")
    
else:
    print(f"\n⏳ Training in progress...")
    
    if best_model.exists():
        print(f"\n✓ Best model saved: {best_model}")
        print(f"  Size: {best_model.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"\n⏸️ No model saved yet")
    
    print(f"\n💡 Check terminal for current progress")

print("\n" + "="*60)
print("\nNext Steps:")
print("  1. Wait for training to complete")
print("  2. Run: python evaluate.py (calculate mAP & FPS)")
print("  3. Run: python test.py (test on sample images)")
print("="*60 + "\n")
