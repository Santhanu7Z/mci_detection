#!/bin/bash
#SBATCH --job-name=mci_xai
#SBATCH --output=logs/xai_%j.out
#SBATCH --error=logs/xai_%j.err
#SBATCH --partition=gpu4070ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "Started at : $(date)"

# ── KEY FIX ───────────────────────────────────────────────────────────────────
# mamba_ssm's Triton layer norm calls torch.cuda.device(x.device.index)
# even on CPU tensors, where x.device.index is None → exchangeDevice(-1) crash.
# MAMBA_NO_TRITON=1 replaces all Triton kernels with plain PyTorch equivalents
# that handle CPU correctly.  Must be set BEFORE mamba_ssm is imported.
export MAMBA_NO_TRITON=1

# Also disable CUDA entirely for XAI — inference-only, CPU is sufficient
export CUDA_VISIBLE_DEVICES=""

source ~/.venv/bin/activate 2>/dev/null || source .venv/bin/activate

cd ~/mci_detection
python run_lodo_xai.py

echo "Finished at: $(date)"

