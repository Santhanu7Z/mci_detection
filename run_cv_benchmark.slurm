#!/bin/bash
#SBATCH --job-name=mamba_fusion_final
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_v4_9_%j.out
#SBATCH --error=logs/benchmark_v4_9_%j.err

# --- Debug Info ---
echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================="

# --- Environment Setup ---
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Ensure logs show up in real-time
export PYTHONUNBUFFERED=1

# --- Execution ---
# Running the v4.9 Unified Benchmark
python benchmark.py

echo "========================================="
echo "Finished:  $(date)"
echo "========================================="
