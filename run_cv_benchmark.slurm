#!/bin/bash
#SBATCH --job-name=mamba_group_cv
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_v3_4_%j.out
#SBATCH --error=logs/benchmark_v3_4_%j.err

# --- Environment Setup ---
# Ensuring the local bin is in path for potential Mamba dependencies
export PATH="$HOME/bin:$PATH"
source .venv/bin/activate

# Environmental Guardrail for Mamba (prevents triton compilation issues on some nodes)
export MAMBA_NO_TRITON=1
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Starting Unified Mamba-Fusion Benchmark v3.4"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "============================================================"

# --- Execution ---
# This runs the full ablation suite: Linguistic, Acoustic, Gated, and Attention Fusion
python benchmark.py

echo "============================================================"
echo "Benchmark Finished: $(date)"
echo "============================================================"
