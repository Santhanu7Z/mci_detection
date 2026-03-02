#!/bin/bash
#SBATCH --job-name=gated_fusion
#SBATCH --partition=gpu3060
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gated_fusion_%j.out
#SBATCH --error=logs/gated_fusion_%j.err

export PATH="$HOME/bin:$PATH"
export MAMBA_NO_TRITON=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

source ~/mci_detection/.venv/bin/activate
cd ~/mci_detection

echo "Training Gated Fusion Model..."
python mamba_gated_fusion.py \
    --epochs 30 \
    --batch_size 8 \
    --lr 2e-4 \
    --acc_steps 4 \
    --patience 8

echo "Gated fusion training complete!"
