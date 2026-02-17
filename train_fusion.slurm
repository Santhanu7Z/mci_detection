#!/bin/bash
#SBATCH --job-name=fusion_train
#SBATCH --partition=gpu4090
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fusion_%j.out
#SBATCH --error=logs/fusion_%j.err

source ~/mci_detection/.venv/bin/activate
cd ~/mci_detection

echo "Training Multimodal Fusion Model..."
python mamba_attention_fusion.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 5e-4

echo "Fusion training complete!"
