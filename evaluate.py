"""
Evaluation and Inference Script
Test trained model on dataset and visualize predictions
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from model import GeoLocalizationModel, haversine_distance, evaluate_model
from dataset import get_dataloaders
import random

# Create output directory
RESULTS_DIR = 'evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_trained_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    model = GeoLocalizationModel(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation distance: {checkpoint['best_val_distance']:.2f} km")
    
    return model, checkpoint


def predict_single_image(model, image_tensor, device):
    """
    Predict coordinates for a single location (3 images)
    
    Args:
        model: Trained model
        image_tensor: Image tensor of shape (3, 3, 224, 224)
        device: torch.device
    
    Returns:
        predicted_lat, predicted_lon
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0).to(device)
        prediction = model(image_batch)
        lat, lon = prediction[0].cpu().numpy()
    
    return lat, lon


def visualize_predictions(model, dataloader, device, num_samples=10, split_name="Test"):
    """
    Visualize model predictions vs actual coordinates
    """
    model.eval()
    
    # Collect predictions
    predictions = []
    actuals = []
    distances = []
    images_data = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            batch_distances = haversine_distance(outputs, targets)
            
            # Store results
            for i in range(len(images)):
                pred_lat, pred_lon = outputs[i].cpu().numpy()
                true_lat, true_lon = targets[i].cpu().numpy()
                dist = batch_distances[i].cpu().item()
                
                predictions.append([pred_lat, pred_lon])
                actuals.append([true_lat, true_lon])
                distances.append(dist)
                images_data.append(images[i].cpu())
            
            # Stop when we have enough samples
            if len(predictions) >= num_samples:
                break
    
    # Trim to exactly num_samples
    predictions = predictions[:num_samples]
    actuals = actuals[:num_samples]
    distances = distances[:num_samples]
    images_data = images_data[:num_samples]
    
    # Create visualization
    # Adjust figure size to accommodate wider images (3 views stitched)
    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    fig.suptitle(f'{split_name} Set Predictions', fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Denormalize images for display and stitch them
        views = []
        for v in range(3):
            img = images_data[idx][v].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            views.append(img)
        
        # Concatenate views horizontally
        full_img = np.concatenate(views, axis=1)
        
        ax.imshow(full_img)
        ax.axis('off')
        
        # Add prediction info
        pred_lat, pred_lon = predictions[idx]
        true_lat, true_lon = actuals[idx]
        dist = distances[idx]
        
        title = f'Error: {dist:.1f} km\n'
        title += f'Pred: ({pred_lat:.2f}, {pred_lon:.2f})\n'
        title += f'True: ({true_lat:.2f}, {true_lon:.2f})'
        
        ax.set_title(title, fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'{split_name.lower()}_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to '{save_path}'")
    plt.close()


def plot_error_distribution(distances, split_name="Test"):
    """Plot histogram of prediction errors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(distances, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.1f} km')
    ax1.axvline(np.median(distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.1f} km')
    ax1.set_xlabel('Distance Error (km)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{split_name} Set - Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
    
    ax2.plot(sorted_distances, cumulative, linewidth=2)
    ax2.axhline(50, color='green', linestyle='--', alpha=0.7, label='50th percentile')
    ax2.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90th percentile')
    ax2.set_xlabel('Distance Error (km)', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title(f'{split_name} Set - Cumulative Error', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'{split_name.lower()}_error_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved error distribution to '{save_path}'")
    plt.close()


def plot_coordinate_scatter(model, dataloader, device, split_name="Test"):
    """Plot predicted vs actual coordinates on a map"""
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot actual locations
    ax.scatter(actuals[:, 1], actuals[:, 0], c='green', s=100, alpha=0.6, 
               marker='o', edgecolors='black', linewidth=1.5, label='Actual')
    
    # Plot predicted locations
    ax.scatter(predictions[:, 1], predictions[:, 0], c='red', s=100, alpha=0.6, 
               marker='x', linewidth=2, label='Predicted')
    
    # Draw lines connecting predictions to actuals
    for i in range(len(predictions)):
        ax.plot([actuals[i, 1], predictions[i, 1]], 
                [actuals[i, 0], predictions[i, 0]], 
                'b-', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{split_name} Set - Predicted vs Actual Locations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set Russia bounds
    ax.set_xlim(20, 180)
    ax.set_ylim(40, 80)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'{split_name.lower()}_coordinate_scatter.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved coordinate scatter to '{save_path}'")
    plt.close()


def evaluate_all_splits(checkpoint_path='checkpoints/best_model.pth'):
    """Evaluate model on all data splits"""
    print("=" * 70)
    print("Model Evaluation and Visualization")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading trained model...")
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return
    
    model, checkpoint = load_trained_model(checkpoint_path, device)
    
    # Load dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        num_workers=4,
        image_size=224
    )
    
    # Evaluate on all splits
    splits = [
        ('Test', test_loader),
        ('Validation', val_loader),
        ('Train', train_loader)
    ]
    
    all_distances = {}
    
    for split_name, loader in splits:
        print(f"\n{'-' * 70}")
        print(f"Evaluating on {split_name} Set")
        print(f"{'-' * 70}")
        
        # Get metrics
        metrics = evaluate_model(model, loader, device)
        
        print(f"\n{split_name} Set Metrics:")
        print(f"  Mean Distance:   {metrics['mean_distance_km']:.2f} km")
        print(f"  Median Distance: {metrics['median_distance_km']:.2f} km")
        print(f"  Min Distance:    {metrics['min_distance_km']:.2f} km")
        print(f"  Max Distance:    {metrics['max_distance_km']:.2f} km")
        print(f"  MSE Loss:        {metrics['mse_loss']:.4f}")
        
        # Collect all distances for this split
        distances = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                batch_distances = haversine_distance(outputs, targets)
                distances.extend(batch_distances.cpu().numpy())
        
        all_distances[split_name] = distances
        
        # Create visualizations
        print(f"\nGenerating visualizations for {split_name} set...")
        visualize_predictions(model, loader, device, num_samples=10, split_name=split_name)
        plot_error_distribution(distances, split_name=split_name)
        plot_coordinate_scatter(model, loader, device, split_name=split_name)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    print(f"\nGenerated visualizations in '{RESULTS_DIR}/' folder:")
    for split_name in ['Test', 'Validation', 'Train']:
        print(f"\n{split_name} Set:")
        print(f"  - {split_name.lower()}_predictions.png")
        print(f"  - {split_name.lower()}_error_distribution.png")
        print(f"  - {split_name.lower()}_coordinate_scatter.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    evaluate_all_splits()
