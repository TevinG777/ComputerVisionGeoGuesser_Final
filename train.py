"""
Training script for GeoGuesser model
Optimized for Google Colab with A100 GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from datetime import datetime
import json

from model import GeoLocalizationModel, get_loss_function, haversine_distance, evaluate_model
from dataset import get_dataloaders


class Trainer:
    """Handles model training with checkpointing and early stopping"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-4,
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Loss and optimizer
        self.criterion = get_loss_function()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create checkpoint directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_distance': [],
            'val_loss': [],
            'val_distance': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_distance = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_distances = []
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate distances for monitoring
            with torch.no_grad():
                distances = haversine_distance(outputs, targets)
                all_distances.extend(distances.cpu().numpy())
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_dist = sum(all_distances) / len(all_distances)
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] - "
                      f"Loss: {avg_loss:.4f}, Avg Distance: {avg_dist:.2f} km")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_distance = sum(all_distances) / len(all_distances)
        
        return avg_loss, avg_distance
    
    def validate(self):
        """Validate the model"""
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_distance': self.best_val_distance,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (Val Distance: {self.best_val_distance:.2f} km)")
    
    def train(self, num_epochs, early_stopping_patience=10):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        print("=" * 70)
        print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Total epochs: {num_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Training
            train_loss, train_distance = self.train_epoch()
            
            # Validation
            print("\n  Validating...")
            val_metrics = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['mean_distance_km'])
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_distance'].append(train_distance)
            self.history['val_loss'].append(val_metrics['mse_loss'])
            self.history['val_distance'].append(val_metrics['mean_distance_km'])
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch Summary:")
            print(f"    Train Loss: {train_loss:.4f} | Train Distance: {train_distance:.2f} km")
            print(f"    Val Loss:   {val_metrics['mse_loss']:.4f} | Val Distance:   {val_metrics['mean_distance_km']:.2f} km")
            print(f"    Val Median: {val_metrics['median_distance_km']:.2f} km | Val Max: {val_metrics['max_distance_km']:.2f} km")
            print(f"    Learning Rate: {current_lr:.2e}")
            print(f"    Time: {epoch_time:.2f}s")
            
            # Check if best model
            is_best = val_metrics['mean_distance_km'] < self.best_val_distance
            if is_best:
                self.best_val_distance = val_metrics['mean_distance_km']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping check
            if self.patience_counter >= early_stopping_patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                print(f"Best model was at epoch {self.best_epoch} with {self.best_val_distance:.2f} km")
                print(f"{'='*70}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation distance: {self.best_val_distance:.2f} km (epoch {self.best_epoch})")
        print(f"Checkpoints saved to: {self.save_dir}")
        
        # Save training history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        print(f"{'='*70}")
        
        return self.history


def main():
    """Main training function"""
    
    # Configuration
    CONFIG = {
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'image_size': 224,
        'num_workers': 4,
        'early_stopping_patience': 10,
        'save_dir': 'checkpoints'
    }
    
    print("=" * 70)
    print("GeoGuesser Model Training")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        image_size=CONFIG['image_size']
    )
    
    # Create model
    print("\nCreating model...")
    model = GeoLocalizationModel(pretrained=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        save_dir=CONFIG['save_dir']
    )
    
    # Train model
    history = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    print("\nTest Set Results:")
    print(f"  MSE Loss: {test_metrics['mse_loss']:.4f}")
    print(f"  Mean Distance: {test_metrics['mean_distance_km']:.2f} km")
    print(f"  Median Distance: {test_metrics['median_distance_km']:.2f} km")
    print(f"  Min Distance: {test_metrics['min_distance_km']:.2f} km")
    print(f"  Max Distance: {test_metrics['max_distance_km']:.2f} km")
    
    # Save test results
    test_results_path = os.path.join(CONFIG['save_dir'], 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nTest results saved to: {test_results_path}")


if __name__ == "__main__":
    main()
