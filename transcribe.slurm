#!/bin/bash
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/transcribe_%j.out
#SBATCH --error=logs/transcribe_%j.err

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

source /storage/slurm/home/santhanu.kuma@cpe.kmutt.ac.th/mci_detection/.venv/bin/activate

which python
python -V

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA device count:', torch.cuda.device_count())"

python create_transcripts.py
