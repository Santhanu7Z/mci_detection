#!/bin/bash
#SBATCH --job-name=mci_xai
#SBATCH --output=logs/xai_%j.out
#SBATCH --error=logs/xai_%j.err
#SBATCH --partition=gpu4070ti
#SBATCH --gres=gpu:1                # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00             # 2 hours should be more than enough

# ── make sure log dir exists ──────────────────────────────────────────────────
mkdir -p logs

# ── print job info ────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "Started at : $(date)"

# ── activate your virtual env ─────────────────────────────────────────────────
source ~/.venv/bin/activate 2>/dev/null || source .venv/bin/activate

# ── run ───────────────────────────────────────────────────────────────────────
cd ~/mci_detection
python explainability_suite.py

echo "Finished at: $(date)"
