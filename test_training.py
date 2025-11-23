"""
Quick test of the training pipeline
Runs 1 epoch to verify everything works together
"""

import torch
from model import GeoLocalizationModel
from dataset import get_dataloaders
from train import Trainer

print("=" * 60)
print("Testing Training Pipeline")
print("=" * 60)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Create small dataloaders for testing (small batch size)
print("\n[1/4] Loading datasets...")
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=8,
    num_workers=0,  # 0 for Windows compatibility
    image_size=224
)
print("✓ Datasets loaded")

# Create model
print("\n[2/4] Creating model...")
model = GeoLocalizationModel(pretrained=False)  # False for faster testing
print("✓ Model created")

# Create trainer
print("\n[3/4] Creating trainer...")
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=1e-4,
    save_dir='test_checkpoints'
)
print("✓ Trainer created")

# Run 1 epoch
print("\n[4/4] Running 1 epoch test...")
print("-" * 60)
history = trainer.train(num_epochs=1, early_stopping_patience=10)
print("-" * 60)

print("\n" + "=" * 60)
print("✅ Training pipeline test successful!")
print("=" * 60)
print("\nReady for full training run!")
print("You can now run: python train.py")
print("\nOr upload to Google Colab for A100 GPU training")
print("=" * 60)
