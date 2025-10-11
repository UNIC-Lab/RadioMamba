#!/usr/bin/env python3
"""
Setup script for RadioMamba project.
Creates necessary directories and prepares the environment.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories for the project."""
    
    # Get the project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    
    # List of directories to create
    directories = [
        'checkpoints/nocars',
        'checkpoints/withcars',
        'results/predictions_nocars',
        'results/predictions_withcars',
        'results/metrics',
        'logs/tensorboard',
        'logs/validation_images_nocars',
        'logs/validation_images_withcars',
        'data',  # For user's dataset
        'tmp'    # For temporary files
    ]
    
    print("Creating necessary directories...")
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\nDirectory structure created successfully!")

def check_dependencies():
    """Check if required packages are installed."""
    
    required_packages = [
        'torch',
        'torchvision', 
        'pytorch_lightning',
        'torchmetrics',
        'numpy',
        'PIL',
        'matplotlib',
        'tqdm',
        'yaml'
    ]
    
    missing_packages = []
    
    print("Checking dependencies...")
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def check_mamba():
    """Check if mamba-ssm is installed (special case as it requires specific installation)."""
    
    print("Checking Mamba SSM...")
    try:
        import mamba_ssm
        print("✓ mamba-ssm is installed")
        return True
    except ImportError:
        print("✗ mamba-ssm is not installed")
        print("Please install it using:")
        print("pip install mamba-ssm")
        print("Note: mamba-ssm requires CUDA and may need specific installation steps.")
        return False

def main():
    """Main setup function."""
    
    print("=" * 50)
    print("RadioMamba Project Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    
    # Check mamba specifically
    mamba_ok = check_mamba()
    
    print("\n" + "=" * 50)
    
    if deps_ok and mamba_ok:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your dataset in the 'data/' directory")
        print("2. Update dataset paths in config files if needed") 
        print("3. Run training: cd src && python train.py --config ../configs/config_nocars.yaml")
    else:
        print("⚠ Setup completed with warnings. Please install missing dependencies.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 