#!/bin/bash
#SBATCH --job-name=whisper_transcribe
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/transcribe_all_%j.out
#SBATCH --error=logs/transcribe_all_%j.err

# --- Debug Info ---
echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Directory: $SLURM_SUBMIT_DIR"
echo "========================================="

# --- Environment Setup ---
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Ensure logs show up in real-time
export PYTHONUNBUFFERED=1

# --- Execution ---
# Using the correct filename 'transcribe_all.py' and specific master paths
python transcribe_all.py \
    --metadata processed_data/master_metadata.csv \
    --output processed_data/transcripts_cache.json \
    --model large-v3

echo "========================================="
echo "Finished:  $(date)"
echo "========================================="
