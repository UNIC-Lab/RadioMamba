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
- 🎯 **Interactive Visualization**: Web-based interactive tool for real-time path loss prediction with click-to-place TX functionality

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
pip install gradio  # Required for interactive visualization
```

> Note: If you encounter issues installing mamba-ssm, refer to https://zhuanlan.zhihu.com/p/27156724975

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

### Interactive Visualization

Launch the interactive web-based visualization tool for real-time path loss prediction:

```bash
cd src
python interactive_visualizer.py
```

**Features:**
- 🖱️ **Click-to-place TX**: Click anywhere on the building map to place a transmitter
- ⚡ **Real-time Prediction**: Instant path loss prediction visualization
- 📍 **Multiple Map Loading**: Select from predefined maps or manually input map number (0-700)
- 📊 **Performance Stats**: Display inference time and coordinate information

**Usage:**
1. The tool will start a Gradio web interface (default: `http://0.0.0.0:7860`)
2. Select a building map from the dropdown or enter a map number manually
3. Click on the building map to place the TX antenna
4. View the predicted path loss distribution on the right panel

**Note:** Make sure to update the `CHECKPOINT_PATH` and `BUILDINGS_DIR` in `interactive_visualizer.py` according to your environment.

**Demo Video:**

Watch the interactive visualization tool in action:

https://drive.google.com/file/d/1nlTFbLkZF-lS5OA2_DPuL86ktiywMsPd/view?usp=drive_link

> 📹 The demo video showcases the interactive visualization tool where you can click on building maps to place TX antennas and get real-time path loss predictions.

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
│   ├── test.py             # Testing script
│   └── interactive_visualizer.py  # Interactive web-based visualization tool
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
@ARTICLE{11190042,
  author={Jia, Honggang and Cheng, Nan and Wang, Xiucheng and Zhou, Conghao and Sun, Ruijin and Shen, Xuemin},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={RadioMamba: Breaking the Accuracy-Efficiency Trade-Off in Radio Map Construction Via a Hybrid Mamba-UNet}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Computational modeling;Accuracy;Real-time systems;Computer architecture;6G mobile communication;Context modeling;Feature extraction;Complexity theory;Wireless networks;Transformers;6 G wireless networks;radio map;Mamba;lightweight model;real-time optimization},
  doi={10.1109/TNSE.2025.3617102}}

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please contact [hgjia@stu.xidian.edu.cn] 
