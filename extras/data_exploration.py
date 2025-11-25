"""
Data Exploration Script for GeoGuesser Dataset
Validates dataset structure and visualizes coordinate distribution
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ANNOTATIONS_DIR = "DataSet/Annotations"
IMAGES_DIR = "DataSet/Images"

def load_coordinates():
    """Load all coordinates from annotation files"""
    coordinates = {}
    annotation_files = sorted([f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('_coords.txt')])
    
    print(f"Found {len(annotation_files)} annotation files")
    
    for ann_file in annotation_files:
        # Extract location ID (e.g., 'image1' from 'image1_coords.txt')
        location_id = ann_file.replace('_coords.txt', '')
        
        # Read coordinates
        with open(os.path.join(ANNOTATIONS_DIR, ann_file), 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                lat = float(lines[0].strip())
                lon = float(lines[1].strip())
                coordinates[location_id] = {'lat': lat, 'lon': lon}
            else:
                print(f"Warning: {ann_file} doesn't have 2 lines")
    
    return coordinates

def verify_images(coordinates):
    """Verify that all expected images exist"""
    missing_images = []
    image_info = []
    
    for location_id in coordinates.keys():
        for angle_idx in range(3):  # Expecting 3 images per location (0, 1, 2)
            img_name = f"{location_id}_{angle_idx}.png"
            img_path = os.path.join(IMAGES_DIR, img_name)
            
            if os.path.exists(img_path):
                # Get image size
                from PIL import Image
                img = Image.open(img_path)
                image_info.append({
                    'name': img_name,
                    'size': img.size,
                    'mode': img.mode
                })
            else:
                missing_images.append(img_name)
    
    return image_info, missing_images

def analyze_coordinates(coordinates):
    """Analyze coordinate distribution"""
    lats = [coord['lat'] for coord in coordinates.values()]
    lons = [coord['lon'] for coord in coordinates.values()]
    
    stats = {
        'num_locations': len(coordinates),
        'lat_min': min(lats),
        'lat_max': max(lats),
        'lat_mean': np.mean(lats),
        'lat_std': np.std(lats),
        'lon_min': min(lons),
        'lon_max': max(lons),
        'lon_mean': np.mean(lons),
        'lon_std': np.std(lons),
    }
    
    return stats, lats, lons

def plot_coordinates(lats, lons):
    """Visualize coordinate distribution"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(lons, lats, alpha=0.6, s=100, c='red', edgecolors='black')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title('Distribution of Locations Across Russia', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add rough Russia borders for context (approximate)
    plt.xlim(20, 180)  # Russia longitude range
    plt.ylim(40, 80)   # Russia latitude range
    
    plt.tight_layout()
    plt.savefig('coordinate_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'coordinate_distribution.png'")
    plt.close()

def check_image_consistency(image_info):
    """Check if all images have consistent dimensions"""
    if not image_info:
        return
    
    sizes = [img['size'] for img in image_info]
    modes = [img['mode'] for img in image_info]
    
    unique_sizes = set(sizes)
    unique_modes = set(modes)
    
    print(f"\nImage dimensions found: {unique_sizes}")
    print(f"Image modes found: {unique_modes}")
    
    if len(unique_sizes) > 1:
        print("⚠️  Warning: Images have different dimensions!")
        size_counts = {}
        for size in sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        for size, count in size_counts.items():
            print(f"  - {size}: {count} images")

def main():
    print("=" * 60)
    print("GeoGuesser Dataset Exploration")
    print("=" * 60)
    
    # Load coordinates
    print("\n[1/5] Loading coordinates...")
    coordinates = load_coordinates()
    print(f"✓ Loaded {len(coordinates)} unique locations")
    
    # Verify images
    print("\n[2/5] Verifying images...")
    image_info, missing_images = verify_images(coordinates)
    print(f"✓ Found {len(image_info)} images")
    
    if missing_images:
        print(f"⚠️  Warning: {len(missing_images)} images are missing:")
        for img in missing_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    else:
        print("✓ All expected images found!")
    
    # Analyze coordinates
    print("\n[3/5] Analyzing coordinate distribution...")
    stats, lats, lons = analyze_coordinates(coordinates)
    
    print("\nCoordinate Statistics:")
    print(f"  Locations: {stats['num_locations']}")
    print(f"  Total Images: {len(image_info)} (expecting {stats['num_locations'] * 3})")
    print(f"\n  Latitude Range:  {stats['lat_min']:.4f} to {stats['lat_max']:.4f}")
    print(f"  Latitude Mean:   {stats['lat_mean']:.4f} ± {stats['lat_std']:.4f}")
    print(f"\n  Longitude Range: {stats['lon_min']:.4f} to {stats['lon_max']:.4f}")
    print(f"  Longitude Mean:  {stats['lon_mean']:.4f} ± {stats['lon_std']:.4f}")
    
    # Check image consistency
    print("\n[4/5] Checking image consistency...")
    check_image_consistency(image_info)
    
    # Visualize
    print("\n[5/5] Creating visualization...")
    plot_coordinates(lats, lons)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Dataset contains {stats['num_locations']} unique locations")
    print(f"✓ Dataset contains {len(image_info)} images")
    print(f"✓ Expected images per location: 3")
    print(f"✓ Coordinates span across Russia")
    
    if len(image_info) == stats['num_locations'] * 3:
        print("\n✅ Dataset is complete and ready for training!")
    else:
        print(f"\n⚠️  Expected {stats['num_locations'] * 3} images but found {len(image_info)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
