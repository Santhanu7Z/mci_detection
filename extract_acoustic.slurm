#!/bin/bash
#SBATCH --job-name=acoustic_extract
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/acoustic_%j.out
#SBATCH --error=logs/acoustic_%j.err

source ~/mci_detection/.venv/bin/activate
cd ~/mci_detection

echo "Extracting acoustic features..."
python acoustic_extractor.py

echo "Preparing fusion data..."
python fusion_data_preprocessor.py

echo "Acoustic feature extraction complete!"
