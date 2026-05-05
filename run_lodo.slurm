#!/bin/bash
# ============================================================
# run_lodo.slurm  —  LODO Benchmark v3.0
# Unfrozen backbone | Max pooling | Composite model selection
# 7 modes × 9 splits × 3 seeds = 189 runs
#
# Runtime estimate (RTX 4090, 12 GB):
#   ~20 epochs × 189 runs × ~3 min/run  ≈  9–12 hours
#   (single-domain small splits finish faster)
# ============================================================
#SBATCH --job-name=lodo_v3
#SBATCH --partition=gpu4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=logs/lodo_%j.out
#SBATCH --error=logs/lodo_%j.err

echo "========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "========================================="

mkdir -p logs

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MAMBA_NO_TRITON=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Prerequisite check
if [ ! -f "processed_data/deep_audio/participant_ids.npy" ]; then
    echo "ERROR: Deep-audio embeddings not found."
    echo "Run:  sbatch run_deep_audio.slurm  first."
    exit 1
fi

# ── Full 189-run matrix ───────────────────────────────────────────────────────
# Safe to resubmit — already-completed runs are skipped automatically.
python lodo_benchmark.py \
    --epochs      20    \
    --min_epochs   8    \
    --patience     5    \
    --batch_size   8    \
    --lr_backbone 2e-5  \
    --lr_head     1e-4  \
    --wd_backbone 0.01  \
    --wd_head     0.05  \
    --max_len     512   \
    --n_seeds      3    \
    --base_seed   42

# ── Optional: run a subset of modes to test first ─────────────────────────────
# e.g. run only acoustic baselines first to verify setup:
#
# python lodo_benchmark.py \
#     --epochs 20 --min_epochs 8 --patience 5 --batch_size 8 \
#     --lr_backbone 2e-5 --lr_head 1e-4 --n_seeds 3 \
#     --modes acoustic_egemaps acoustic_w2v acoustic_hubert

echo "========================================="
echo "Finished : $(date)"
echo "========================================="
