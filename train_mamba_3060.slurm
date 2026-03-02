#!/bin/bash
#SBATCH --job-name=mamba_train
#SBATCH --partition=gpu3060
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_mamba_%j.out
#SBATCH --error=logs/train_mamba_%j.err

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "========================================="

# Set PATH to include ffmpeg
export PATH="$HOME/bin:$PATH"

# Activate virtual environment
source ~/mci_detection/.venv/bin/activate

# Navigate to project directory
cd ~/mci_detection

# Verify GPU is available
echo "Checking GPU availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# Only run preprocessing if not already done
if [ ! -f "processed_data/metadata.csv" ]; then
    echo "Starting preprocessing..."
    python preprocess.py
else
    echo "Preprocessing already done, skipping..."
fi

# Train Mamba model
echo "Training Mamba text-only model..."
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 5e-4 \
    --patience 7 \
    --max_len 512

echo "========================================="
echo "Training completed!"
echo "Finished: $(date)"
echo "========================================="
