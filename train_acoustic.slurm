#!/bin/bash
#SBATCH --job-name=acoustic_train
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/acoustic_%j.out
#SBATCH --error=logs/acoustic_%j.err

export PATH="$HOME/bin:$PATH"
source ~/mci_detection/.venv/bin/activate
cd ~/mci_detection

# COMMENTED OUT: We already did this in Job 469!
# echo "Extracting acoustic features..."
# python acoustic_extractor.py

# COMMENTED OUT: We assume this is already done, or you can uncomment if needed.
# echo "Preparing fusion data..."
# python fusion_data_preprocessor.py

echo "Training acoustic-only baseline..."
# Passing the optimized hyperparameters we discussed
python acoustic_only.py --epochs 40 --batch_size 16 --lr 3e-4

echo "Acoustic training complete!"
