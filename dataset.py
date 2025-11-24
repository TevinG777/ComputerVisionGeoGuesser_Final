"""
PyTorch Dataset for GeoGuesser
Loads images and coordinates with location-based train/val/test split
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import random


class GeoGuesserDataset(Dataset):
    """
    Dataset for geo-location prediction from images.
    Each location has 3 images from different angles of the same 360° panorama.
    """
    
    def __init__(self, split='train', transform=None, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: torchvision transforms to apply to images
            train_ratio: proportion of locations for training (default 0.7)
            val_ratio: proportion of locations for validation (default 0.15)
            seed: random seed for reproducible splits
        """
        self.split = split
        self.transform = transform
        self.annotations_dir = "DataSet/Annotations"
        self.images_dir = "DataSet/Images"
        
        # Load all location data
        self.data = self._load_data()
        
        # Split data by location (not by individual images)
        self.data = self._split_data(train_ratio, val_ratio, seed)
        
        print(f"{split.upper()} set: {len(self.data)} images from {len(set([d['location_id'] for d in self.data]))} locations")
    
    def _load_data(self):
        """Load all images and their coordinates, grouped by location"""
        data = []
        
        # Get all annotation files
        annotation_files = sorted([f for f in os.listdir(self.annotations_dir) 
                                  if f.endswith('_coords.txt')])
        
        for ann_file in annotation_files:
            # Extract location ID (e.g., 'image1' from 'image1_coords.txt')
            location_id = ann_file.replace('_coords.txt', '')
            
            # Read coordinates
            with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
                lines = f.readlines()
                lat = float(lines[0].strip())
                lon = float(lines[1].strip())
            
            # Check if all 3 images exist for this location
            image_paths = []
            all_images_exist = True
            
            for angle_idx in range(3):
                img_name = f"{location_id}_{angle_idx}.png"
                img_path = os.path.join(self.images_dir, img_name)
                
                if not os.path.exists(img_path):
                    all_images_exist = False
                    break
                
                image_paths.append(img_path)
            
            # Only add location if all 3 images exist
            if all_images_exist:
                data.append({
                    'image_paths': image_paths,
                    'lat': lat,
                    'lon': lon,
                    'location_id': location_id
                })
        
        return data
    
    def _split_data(self, train_ratio, val_ratio, seed):
        """
        Split data by location.
        Since self.data is already grouped by location, we just split the list.
        """
        # Shuffle locations with seed
        random.seed(seed)
        # Create a copy to avoid shuffling the original if called multiple times
        shuffled_data = self.data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
        num_locations = len(shuffled_data)
        train_end = int(num_locations * train_ratio)
        val_end = train_end + int(num_locations * val_ratio)
        
        # Split data
        if self.split == 'train':
            return shuffled_data[:train_end]
        elif self.split == 'val':
            return shuffled_data[train_end:val_end]
        elif self.split == 'test':
            return shuffled_data[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            images: torch.Tensor of shape (3, 3, H, W) - (Views, Channels, Height, Width)
            target: torch.Tensor of shape (2,) containing [lat, lon]
        """
        item = self.data[idx]
        
        images = []
        for img_path in item['image_paths']:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
        
        # Stack images along a new dimension
        # Result shape: (3, 3, H, W)
        images_stack = torch.stack(images)
        
        # Prepare target (lat, lon)
        target = torch.tensor([item['lat'], item['lon']], dtype=torch.float32)
        
        return images_stack, target


def get_transforms(split='train', image_size=224):
    """
    Get appropriate transforms for each split.
    
    Training transforms simulate the game conditions:
    - Random crops (simulates zoom in/out)
    - Small rotations (simulates camera tilt)
    - Color jitter (simulates lighting variations)
    - Gaussian blur (simulates rendering quality)
    
    Validation/Test transforms are minimal (just resize and normalize).
    """
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Slightly larger for random crop
            transforms.RandomCrop(image_size),  # Simulates zoom
            transforms.RandomRotation(degrees=5),  # Small tilt
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(batch_size=32, num_workers=4, image_size=224, train_ratio=0.7, val_ratio=0.15):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        batch_size: number of samples per batch
        num_workers: number of worker processes for data loading
        image_size: size to resize images to (default 224 for ResNet)
        train_ratio: proportion of locations for training
        val_ratio: proportion of locations for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = GeoGuesserDataset(
        split='train',
        transform=get_transforms('train', image_size),
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    val_dataset = GeoGuesserDataset(
        split='val',
        transform=get_transforms('val', image_size),
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    test_dataset = GeoGuesserDataset(
        split='test',
        transform=get_transforms('test', image_size),
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset and dataloaders"""
    print("=" * 60)
    print("Testing GeoGuesser Dataset")
    print("=" * 60)
    
    # Test dataset creation
    print("\n[1/3] Creating datasets...")
    train_dataset = GeoGuesserDataset(split='train', transform=get_transforms('train'))
    val_dataset = GeoGuesserDataset(split='val', transform=get_transforms('val'))
    test_dataset = GeoGuesserDataset(split='test', transform=get_transforms('test'))
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} images")
    
    # Test data loading
    print("\n[2/3] Testing data loading...")
    images, target = train_dataset[0]
    print(f"✓ Images shape: {images.shape} (should be [3, 3, H, W])")
    print(f"✓ Target shape: {target.shape}")
    print(f"✓ Target (lat, lon): [{target[0]:.4f}, {target[1]:.4f}]")
    
    # Test dataloaders
    print("\n[3/3] Testing dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    # Get a batch
    images, targets = next(iter(train_loader))
    print(f"✓ Batch images shape: {images.shape} (should be [B, 3, 3, H, W])")
    print(f"✓ Batch targets shape: {targets.shape}")
    print(f"✓ Batch size: {images.shape[0]}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Dataset is ready for training.")
    print("=" * 60)
