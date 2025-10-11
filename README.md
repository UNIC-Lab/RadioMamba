# 🚀 RadioMamba: Radio Map Prediction using Mamba Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

> 🛰️ **RadioMamba** is a deep learning framework for radio map prediction, combining the power of Mamba state space models with U-Net for efficient and accurate radio propagation modeling.

---

## ✨ Features

- 🧬 **Hybrid Architecture**: Combines Mamba state space models with convolutional layers for global and local feature extraction
- 🚗 **Multi-scenario Support**: Supports both scenarios with and without vehicle obstacles
- ⚙️ **Flexible Configuration**: YAML-based configuration system for easy experimentation
- 📊 **Comprehensive Evaluation**: Built-in metrics including MSE, PSNR, SSIM, and NMSE
- ⚡ **Lightning Integration**: Built on PyTorch Lightning for scalable training

---

## 🏗️ Architecture

RadioMamba implements a hybrid U-Net architecture featuring:
- **MambaConvBlock**: Core building block combining SS2D Mamba for global context and ResidualConvBlock for local features
- **Multi-scale Processing**: Encoder-decoder structure with skip connections
- **Efficient Training**: Supports mixed precision and distributed training

## Installation

### Requirements

Install the exact versions used in development:

```bash
pip install -r requirements.txt
```

Or install individual packages:
```bash
pip install torch torchvision pytorch-lightning
pip install mamba-ssm
pip install torchmetrics
pip install pillow numpy matplotlib tqdm pyyaml
```

Verify your installation:
```bash
python scripts/verify_requirements.py
```

### Clone Repository

```bash
git clone https://github.com/your-username/RadioMamba.git
cd RadioMamba
```

## Usage

### Training

Train the model with cars scenario:
```bash
cd src
python train.py --config ../configs/config_withcars.yaml
```

Train the model without cars:
```bash
cd src
python train.py --config ../configs/config_nocars.yaml
```

### Testing

Generate predictions:
```bash
cd src
python test.py --config ../configs/config_withcars.yaml
```

### Evaluation

Evaluate model performance:
```bash
cd evaluation
python evaluate_withcars.py  # For scenarios with cars
python evaluate_nocars.py    # For scenarios without cars
```

## Configuration

The project uses YAML configuration files located in the `configs/` directory:

- `config_nocars.yaml`: Configuration for scenarios without vehicles
- `config_withcars.yaml`: Configuration for scenarios with vehicles

Key configuration sections:
- **Model**: Architecture parameters (dimensions, depths, Mamba settings)
- **Training**: Learning rate, loss weights, optimization settings
- **Data**: Dataset paths, batch size, data loading parameters
- **Logging**: TensorBoard and validation image logging settings

## Project Structure

```
RadioMamba/
├── src/                     # Source code
│   ├── model.py            # RadioMamba model definition
│   ├── dataset.py          # Data loading utilities
│   ├── train.py            # Training script
│   └── test.py             # Testing script
├── configs/                # Configuration files
│   ├── config_nocars.yaml  # No cars scenario config
│   └── config_withcars.yaml # With cars scenario config
├── evaluation/             # Evaluation scripts
│   ├── evaluate_nocars.py  # Evaluation for no cars
│   └── evaluate_withcars.py # Evaluation with cars
├── logs/                   # Training logs and validation images
│   ├── validation_images_nocars/
│   ├── validation_images_withcars/
│   └── tensorboard/
├── scripts/                # Utility scripts
│   ├── setup.py           # Environment setup
│   └── quick_start.sh     # Quick start script
├── docs/                   # Documentation
└── README.md
```

## Model Architecture Details

### RadioMambaNet
- **Input Channels**: 3 (buildings, transmitters, cars/buildings)
- **Output Channels**: 1 (path loss prediction)
- **Encoder**: 4 stages with increasing dimensions [48, 96, 192, 384]
- **Decoder**: Progressive upsampling with skip connections
- **Loss Function**: Combined loss (L1 + MSE + SSIM + Gradient)

### MambaConvBlock
Each block contains:
- **SS2D Mamba Branch**: For global spatial dependencies
- **Residual Conv Branch**: For local feature extraction
- **Feature Fusion**: Element-wise addition of both branches

## Training Details

- **Optimizer**: AdamW with learning rate scheduling
- **Batch Size**: 25 per GPU (50 total on 2 GPUs)
- **Precision**: Mixed precision (16-bit)
- **Early Stopping**: Based on validation loss with patience
- **Checkpointing**: Best and latest model saving

## Evaluation Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **NMSE**: Normalized Mean Squared Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

## Dataset

The model expects RadioMapSeer dataset format:
- Building maps (grayscale images)
- Transmitter location maps
- Optional: Vehicle/car distribution maps
- Ground truth: DPM path loss maps

## Citation

If you use RadioMamba in your research, please cite:

```bibtex
@article{radiomama2024,
  title={RadioMamba: Radio Map Prediction using Mamba Architecture},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please contact [your-email@example.com] 