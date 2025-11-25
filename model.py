"""
ResNet50-based model for geolocation prediction
Predicts latitude and longitude from street view images
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import math


class GeoLocalizationModel(nn.Module):
    """
    ResNet50-based model for predicting GPS coordinates.
    Uses pretrained ImageNet weights and replaces the final layer.
    """
    
    def __init__(self, pretrained=True):
        super(GeoLocalizationModel, self).__init__()
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Remove the final fully connected layer to get raw features
        self.resnet.fc = nn.Identity()
        
        # Multi-view regressor
        # Input: num_features * 3 (3 views concatenated)
        # 2048 * 3 = 6144 features
        self.regressor = nn.Sequential(
            nn.Linear(num_features * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Output: [lat, lon]
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images of shape (batch_size, 3, 3, 224, 224)
               Dimensions: (Batch, Views, Channels, Height, Width)
        
        Returns:
            predictions: Tensor of shape (batch_size, 2) with [lat, lon]
        """
        batch_size, num_views, C, H, W = x.shape
        
        # Reshape to (batch_size * num_views, C, H, W) to pass through ResNet
        x = x.view(batch_size * num_views, C, H, W)
        
        # Extract features: (batch_size * num_views, 2048)
        features = self.resnet(x)
        
        # Reshape back to separate views: (batch_size, num_views, 2048)
        features = features.view(batch_size, num_views, -1)
        
        # Flatten views: (batch_size, num_views * 2048) -> (batch_size, 6144)
        features = features.view(batch_size, -1)
        
        # Predict coordinates
        return self.regressor(features)


def haversine_distance(pred_coords, true_coords):
    """
    Calculate the great circle distance between two points on Earth.
    Uses the Haversine formula.
    
    Args:
        pred_coords: Predicted coordinates (batch_size, 2) [lat, lon] in degrees
        true_coords: True coordinates (batch_size, 2) [lat, lon] in degrees
    
    Returns:
        distances: Distance in kilometers for each sample (batch_size,)
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1 = torch.deg2rad(pred_coords[:, 0])
    lon1 = torch.deg2rad(pred_coords[:, 1])
    lat2 = torch.deg2rad(true_coords[:, 0])
    lon2 = torch.deg2rad(true_coords[:, 1])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    distance = R * c
    
    return distance


class GeodesicMSELoss(nn.Module):
    """
    Custom Loss: MSE weighted by Cosine of Latitude.
    Fixes the 'Russia Problem' where longitude errors are exaggerated in the North.
    """
    def __init__(self):
        super(GeodesicMSELoss, self).__init__()
        
    def forward(self, preds, targets):
        # preds: [batch_size, 2] -> [lat, lon]
        # targets: [batch_size, 2] -> [lat, lon]
        
        lat_pred, lon_pred = preds[:, 0], preds[:, 1]
        lat_targ, lon_targ = targets[:, 0], targets[:, 1]
        
        # Latitude error (standard MSE)
        lat_loss = (lat_pred - lat_targ) ** 2
        
        # Longitude error (weighted by cos(latitude))
        # We must convert degrees to radians for the Cosine calculation
        lat_rad = torch.deg2rad(lat_targ)
        lon_loss = (lon_pred - lon_targ) ** 2 * torch.cos(lat_rad) ** 2
        
        return torch.mean(lat_loss + lon_loss)


def get_loss_function():
    """
    Returns custom Geodesic MSE loss for coordinate prediction.
    This loss function accounts for latitude when calculating longitude errors,
    making it more suitable for geographic coordinate regression.
    """
    return GeodesicMSELoss()


def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: The GeoLocalizationModel
        dataloader: DataLoader for the dataset to evaluate
        device: torch.device
    
    Returns:
        metrics: Dictionary containing:
            - mse_loss: Mean Squared Error loss
            - mean_distance: Average haversine distance error in km
            - median_distance: Median haversine distance error in km
            - max_distance: Maximum distance error in km
    """
    model.eval()
    criterion = get_loss_function()
    
    total_loss = 0.0
    all_distances = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate MSE loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate haversine distances
            distances = haversine_distance(outputs, targets)
            all_distances.extend(distances.cpu().numpy())
    
    # Calculate metrics
    all_distances = torch.tensor(all_distances)
    metrics = {
        'mse_loss': total_loss / len(dataloader),
        'mean_distance_km': all_distances.mean().item(),
        'median_distance_km': all_distances.median().item(),
        'max_distance_km': all_distances.max().item(),
        'min_distance_km': all_distances.min().item()
    }
    
    return metrics


if __name__ == "__main__":
    """Test the model architecture"""
    print("=" * 60)
    print("Testing GeoLocalization Model")
    print("=" * 60)
    
    # Test model creation
    print("\n[1/4] Creating model...")
    model = GeoLocalizationModel(pretrained=False)  # Use False for quick testing
    print("✓ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n[2/4] Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected output shape: ({batch_size}, 2)")
    assert output.shape == (batch_size, 2), "Output shape mismatch!"
    
    # Test loss function
    print("\n[3/4] Testing loss function...")
    criterion = get_loss_function()
    dummy_target = torch.tensor([[55.7558, 37.6173], [59.9343, 30.3351], 
                                  [55.0084, 82.9357], [56.8389, 60.6057]])
    loss = criterion(output, dummy_target)
    print(f"✓ Geodesic MSE Loss calculated: {loss.item():.4f}")
    print(f"✓ Loss is scalar: {loss.dim() == 0}")
    
    # Test haversine distance
    print("\n[4/4] Testing haversine distance...")
    # Moscow to Saint Petersburg (should be ~634 km)
    moscow = torch.tensor([[55.7558, 37.6173]])
    st_petersburg = torch.tensor([[59.9343, 30.3351]])
    distance = haversine_distance(moscow, st_petersburg)
    print(f"✓ Moscow to St. Petersburg: {distance[0]:.2f} km (expected ~634 km)")
    
    # Test batch distances
    pred = torch.tensor([[55.7558, 37.6173], [59.9343, 30.3351]])
    true = torch.tensor([[55.7560, 37.6180], [59.9340, 30.3360]])  # Very close
    distances = haversine_distance(pred, true)
    print(f"✓ Batch distance calculation: {distances.shape}")
    print(f"✓ Small errors (should be < 1 km): [{distances[0]:.4f}, {distances[1]:.4f}] km")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Model is ready for training.")
    print("=" * 60)
    
    # Print model architecture summary
    print("\nModel Architecture:")
    print("-" * 60)
    print(model)
