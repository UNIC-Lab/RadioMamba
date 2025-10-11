#!/usr/bin/env python3
"""
Verify that all required packages are installed with correct versions.
This script checks if the environment matches the requirements.txt file.
"""

import sys
import importlib

# Required packages with their expected versions
REQUIRED_PACKAGES = {
    'torch': '2.2.0+cu118',
    'torchvision': '0.17.0+cu118', 
    'pytorch_lightning': '2.5.1.post0',
    'mamba_ssm': '1.1.3',
    'torchmetrics': '1.7.1',
    'PIL': '11.2.1',  # Pillow
    'numpy': '1.26.4',
    'skimage': '0.25.2',  # scikit-image
    'matplotlib': '3.10.3',
    'tensorboard': '2.19.0',
    'tqdm': '4.67.1',
    'yaml': '6.0.2',  # PyYAML
}

# Package name mappings for display
PACKAGE_DISPLAY_NAMES = {
    'PIL': 'Pillow',
    'skimage': 'scikit-image',
    'yaml': 'PyYAML',
    'pytorch_lightning': 'pytorch-lightning',
    'mamba_ssm': 'mamba-ssm'
}

def check_package(package_name, expected_version):
    """Check if a package is installed with the expected version."""
    try:
        # Import the package
        if package_name == 'PIL':
            import PIL as module
        elif package_name == 'skimage':
            import skimage as module
        elif package_name == 'yaml':
            import yaml as module
        else:
            module = importlib.import_module(package_name)
        
        # Get version
        actual_version = module.__version__
        
        # Display name
        display_name = PACKAGE_DISPLAY_NAMES.get(package_name, package_name)
        
        # Check version match
        if actual_version == expected_version:
            print(f"✓ {display_name}: {actual_version} (OK)")
            return True
        else:
            print(f"⚠ {display_name}: {actual_version} (Expected: {expected_version})")
            return False
            
    except ImportError:
        display_name = PACKAGE_DISPLAY_NAMES.get(package_name, package_name)
        print(f"✗ {display_name}: Not installed")
        return False
    except AttributeError:
        display_name = PACKAGE_DISPLAY_NAMES.get(package_name, package_name)
        print(f"? {display_name}: No version info available")
        return False

def main():
    """Main verification function."""
    print("=" * 60)
    print("RadioMamba Requirements Verification")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for package, expected_version in REQUIRED_PACKAGES.items():
        if check_package(package, expected_version):
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All required packages are correctly installed!")
        print("\nYou can now run:")
        print("  python scripts/setup.py  # Setup directories")
        print("  cd src && python train.py --config ../configs/config_nocars.yaml")
    else:
        print("⚠ Some packages are missing or have different versions.")
        print("\nTo install the exact versions, run:")
        print("  pip install -r requirements.txt")
        print("\nNote: This may downgrade/upgrade some packages.")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 