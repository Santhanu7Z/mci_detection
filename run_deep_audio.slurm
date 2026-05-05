#!/bin/bash
# ============================================================
# run_deep_audio.slurm  — Stage 4c: Wav2Vec2 + HuBERT extraction
# ============================================================
#SBATCH --job-name=deep_audio
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/deep_audio_%j.out
#SBATCH --error=logs/deep_audio_%j.err

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================="

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python deep_audio_extractor.py \
    --meta    processed_data/master_metadata.csv \
    --output  processed_data/deep_audio

echo "========================================="
echo "Finished:  $(date)"
echo "========================================="
