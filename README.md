# GeoGuesser Model - Russia Location Prediction

A ResNet50-based deep learning model that predicts GPS coordinates (latitude, longitude) from street view images across Russia.

## Overview

This model is trained on 130 unique locations (390 images total) with 3 different viewing angles per location from 360° panoramas. The model learns to predict coordinates across 10-15 geographic clusters throughout Russia.

## Dataset Structure

```
DataSet/
├── Images/           # 390 images (130 locations × 3 angles each)
│   ├── image1_0.png
│   ├── image1_1.png
│   ├── image1_2.png
│   └── ...
└── Annotations/      # Coordinate files
    ├── image1_coords.txt  (format: latitude\nlongitude)
    └── ...
```

## Installation

### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Explore the Dataset

```bash
python data_exploration.py
```

This will:
- Validate all 130 locations and 390 images
- Show coordinate statistics
- Generate a visualization map (`coordinate_distribution.png`)

### 2. Test the Dataset and Model

```bash
# Test dataset loading
python dataset.py

# Test model architecture
python model.py
```

### 3. Train the Model

**Quick test (1 epoch):**
```bash
python test_training.py
```

**Full training:**
```bash
python train.py
```

Training configuration:
- Batch size: 32
- Learning rate: 1e-4 (with ReduceLROnPlateau scheduler)
- Early stopping: 10 epochs patience
- Data split: 70% train / 15% val / 15% test (location-based)
- Checkpoints saved to `checkpoints/`

### 4. Google Colab Training (Recommended for A100 GPU)

Upload your dataset and code to Google Colab for faster training with GPU acceleration.

## Model Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Custom Head**: 2048 → 512 → 128 → 2 (with dropout)
- **Output**: [latitude, longitude]
- **Loss**: MSE Loss
- **Evaluation Metric**: Haversine distance (km)

## Data Augmentation

Training augmentations simulate real game conditions:
- Random crops (zoom in/out)
- Small rotations (±5°, camera tilt)
- Color jitter (lighting variations)
- Gaussian blur (rendering quality)

**No horizontal flips** to preserve geographic features.

## Data Split Strategy

Uses **location-based splitting** (Option A - Conservative):
- All 3 images from the same location stay together in the same split
- Prevents data leakage
- Better tests generalization to truly new locations
- Split: ~91 train / ~19 val / ~20 test locations

## Results

After training, the model checkpoints and metrics are saved:
- `checkpoints/best_model.pth` - Best model based on validation distance
- `checkpoints/latest_checkpoint.pth` - Latest epoch checkpoint
- `checkpoints/training_history.json` - Training metrics over time
- `checkpoints/test_results.json` - Final test set evaluation

## Project Structure

```
.
├── data_exploration.py    # Dataset validation and visualization
├── dataset.py            # PyTorch Dataset and DataLoader
├── model.py              # ResNet50 model architecture
├── train.py              # Training script
├── test_training.py      # Quick training test
├── pyproject.toml        # Poetry dependencies
├── README.md             # This file
└── DataSet/
    ├── Images/
    └── Annotations/
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 4GB+ RAM
- ~10GB disk space

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests!
