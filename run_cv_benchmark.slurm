#!/bin/bash
#SBATCH --job-name=cv_benchmark
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cv_benchmark_%j.out
#SBATCH --error=logs/cv_benchmark_%j.err

export PATH="$HOME/bin:$PATH"
export MAMBA_NO_TRITON=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

source ~/mci_detection/.venv/bin/activate
cd ~/mci_detection

python -u benchmark.py

