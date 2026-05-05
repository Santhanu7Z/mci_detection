#!/bin/bash
#SBATCH --job-name=mci_severity
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00        # Severity training is faster than LODO; 6 hours is plenty
#SBATCH --output=logs/severity_%j.out
#SBATCH --error=logs/severity_%j.err

# 1. Navigate to project
cd ~/mci_detection

# 2. Activate environment
source .venv/bin/activate

# 3. Run the ordinal severity experiment
# Using 5-fold cross-validation and 30 epochs
python severity_grader.py --epochs 30 --batch_size 8 --folds 5 --seed 42
