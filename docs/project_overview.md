# RadioMamba Project Overview

## Project Reorganization Summary

This document provides an overview of the RadioMamba project after reorganization for GitHub release.

## What is RadioMamba?

RadioMamba is a deep learning framework for radio map prediction that combines:
- **Mamba State Space Models**: For efficient long-range dependency modeling
- **U-Net Architecture**: For spatial feature extraction and reconstruction
- **Multi-scenario Support**: Works with and without vehicle obstacles

## Key Components

### 1. Core Architecture
- **RadioMambaNet**: Hybrid model combining Mamba SSM and CNN
- **MambaConvBlock**: Core building block with parallel Mamba and Conv branches
- **Multi-scale Processing**: Encoder-decoder with skip connections

### 2. Training Features
- **Combined Loss Function**: L1 + MSE + SSIM + Gradient loss
- **Mixed Precision Training**: 16-bit for efficiency
- **Distributed Training**: Multi-GPU support with PyTorch Lightning
- **Advanced Callbacks**: Early stopping, checkpointing, validation visualization

### 3. Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- NMSE (Normalized Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

## Project Structure

```
RadioMamba/
├── src/                    # Source code
│   ├── model.py           # RadioMambaNet implementation
│   ├── dataset.py         # Data loading utilities
│   ├── train.py           # Training script
│   └── test.py            # Testing/inference script
├── configs/               # Configuration files
│   ├── config_nocars.yaml     # No vehicles scenario
│   └── config_withcars.yaml   # With vehicles scenario
├── evaluation/            # Evaluation tools
│   ├── evaluate_nocars.py     # Metrics for no cars
│   └── evaluate_withcars.py   # Metrics with cars
├── scripts/               # Utility scripts
│   ├── setup.py              # Environment setup
│   ├── quick_start.sh         # Quick start guide
│   └── verify_requirements.py # Dependency verification
├── logs/                  # Training logs (samples)
│   ├── validation_images_nocars/
│   ├── validation_images_withcars/
│   └── tensorboard/
└── docs/                  # Documentation
```

## Model Specifications

### RadioMambaNet v14 Configuration
- **Input Channels**: 3 (buildings, transmitters, cars/buildings)
- **Output Channels**: 1 (path loss prediction)
- **Model Size**: ~8.97M parameters
- **Architecture Dims**: [48, 96, 192, 384]
- **Encoder Depths**: [2, 3, 4, 2]
- **Mamba Config**: d_state=32, d_conv=4, expand=2

### Training Configuration
- **Optimizer**: AdamW (lr=0.0005, weight_decay=0.0001)
- **Batch Size**: 25 per GPU (50 total on 2 GPUs)
- **Loss Weights**: L1=0.4, MSE=0.1, SSIM=0.2, Gradient=0.3
- **Precision**: Mixed 16-bit
- **Early Stopping**: Patience=14 validation checks

## Quick Start

### 1. Setup Environment
```bash
# Install exact dependency versions
pip install -r requirements.txt

# Verify installation
python scripts/verify_requirements.py

# Setup project directories
python scripts/setup.py
```

### 2. Training
```bash
# Train without cars
cd src && python train.py --config ../configs/config_nocars.yaml

# Train with cars
cd src && python train.py --config ../configs/config_withcars.yaml
```

### 3. Testing
```bash
# Generate predictions
cd src && python test.py --config ../configs/config_nocars.yaml
```

### 4. Evaluation
```bash
# Evaluate predictions
cd evaluation && python evaluate_nocars.py
```

## Changes from Original Project

### File Reorganization
- Removed version numbers from file names
- Standardized directory structure for GitHub
- Updated import paths for new structure
- Centralized configuration files

### Path Updates
- Changed absolute paths to relative paths
- Updated checkpoint and log directories
- Simplified result directory structure

### Documentation
- Added comprehensive README
- Created setup and quick start scripts
- Included project overview documentation

## Dataset Requirements

The model expects RadioMapSeer dataset format:
- **Building Maps**: Grayscale images of building layouts
- **Transmitter Maps**: Binary maps showing transmitter locations
- **Vehicle Maps**: (Optional) Binary maps showing vehicle distributions
- **Ground Truth**: DPM path loss maps (target outputs)

## Performance Notes

### Model Performance (v14)
- **Without Cars**: Best validation loss ~0.0125
- **With Cars**: Best validation loss ~0.0156
- **Training Time**: ~26k-29k steps to convergence
- **Inference Speed**: ~0.001s per image (GPU)

### Hardware Requirements
- **GPU**: CUDA-capable GPU (recommended: RTX 3080 or better)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: 50GB+ for full dataset and logs

## Key Features for GitHub Release

1. **Clean Structure**: Organized for easy navigation and contribution
2. **Documentation**: Comprehensive README and guides
3. **Automation**: Setup and quick start scripts
4. **Configuration**: Flexible YAML-based configuration
5. **Reproducibility**: Fixed seeds and detailed specifications

## Next Steps

1. Update dataset paths in configuration files
2. Test the reorganized structure
3. Add any missing documentation
4. Prepare for GitHub release
5. Consider adding CI/CD workflows 