#!/bin/bash

# RadioMamba Quick Start Script
# This script helps users quickly set up and start training

echo "========================================"
echo "RadioMamba Quick Start"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the RadioMamba project root directory"
    exit 1
fi

# Run setup
echo "Running setup..."
python scripts/setup.py

# Check exit status
if [ $? -ne 0 ]; then
    echo "Setup failed. Please check the errors above."
    exit 1
fi

echo ""
echo "========================================"
echo "Quick Start Options"
echo "========================================"
echo "1. Train model without cars"
echo "2. Train model with cars"
echo "3. Test model (requires existing checkpoint)"
echo "4. Evaluate model predictions"
echo "5. Exit"
echo ""

read -p "Select an option (1-5): " choice

case $choice in
    1)
        echo "Starting training without cars..."
        cd src
        python train.py --config ../configs/config_nocars.yaml
        ;;
    2)
        echo "Starting training with cars..."
        cd src
        python train.py --config ../configs/config_withcars.yaml
        ;;
    3)
        echo "Available config files:"
        echo "1. config_nocars.yaml"
        echo "2. config_withcars.yaml"
        read -p "Select config (1-2): " config_choice
        
        if [ "$config_choice" = "1" ]; then
            config_file="../configs/config_nocars.yaml"
        elif [ "$config_choice" = "2" ]; then
            config_file="../configs/config_withcars.yaml"
        else
            echo "Invalid choice"
            exit 1
        fi
        
        echo "Starting testing..."
        cd src
        python test.py --config $config_file
        ;;
    4)
        echo "Available evaluation scripts:"
        echo "1. Evaluate without cars"
        echo "2. Evaluate with cars"
        read -p "Select evaluation (1-2): " eval_choice
        
        if [ "$eval_choice" = "1" ]; then
            eval_script="evaluate_nocars.py"
        elif [ "$eval_choice" = "2" ]; then
            eval_script="evaluate_withcars.py"
        else
            echo "Invalid choice"
            exit 1
        fi
        
        echo "Starting evaluation..."
        cd evaluation
        python $eval_script
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Operation completed!"
echo "========================================" 