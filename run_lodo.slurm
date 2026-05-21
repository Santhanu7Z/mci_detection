#!/bin/bash
# ============================================================
# run_lodo.slurm  —  LODO Benchmark v4.1
# Handles both baseline and ablation runs from a single file.
#
# USAGE
# ─────
# Baseline (default):
#   sbatch run_lodo.slurm
#
# Ablation (custom feature CSV + tag):
#   sbatch --export=FEATURE_CSV=master_acoustic_features_robust.csv,RUN_TAG=ablation run_lodo.slurm
#
# Both jobs can run simultaneously — output files never collide.
# ============================================================
#SBATCH --job-name=lodo_v4
#SBATCH --partition=gpu4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=logs/lodo_%x_%j.out
#SBATCH --error=logs/lodo_%x_%j.err

echo "========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
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

# ── Defaults (used when no --export is passed) ────────────────────────────────
FEATURE_CSV=${FEATURE_CSV:-master_acoustic_features.csv}
RUN_TAG=${RUN_TAG:-}

# ── Print run config so the log is self-documenting ──────────────────────────
echo "Feature CSV : $FEATURE_CSV"
echo "Run tag     : ${RUN_TAG:-<none, baseline>}"
echo "========================================="

# ── Prerequisite check ────────────────────────────────────────────────────────
if [ ! -f "processed_data/deep_audio/participant_ids.npy" ]; then
    echo "ERROR: Deep-audio embeddings not found."
    echo "Run:  sbatch run_deep_audio.slurm  first."
    exit 1
fi

# ── Build the python command dynamically ─────────────────────────────────────
CMD="python lodo_benchmark.py \
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
    --base_seed   42    \
    --feature_csv $FEATURE_CSV"

# Only append --run_tag if one was provided
if [ -n "$RUN_TAG" ]; then
    CMD="$CMD --run_tag $RUN_TAG"
fi

echo "Command: $CMD"
echo "========================================="
eval $CMD

echo "========================================="
echo "Finished : $(date)"
echo "========================================="
