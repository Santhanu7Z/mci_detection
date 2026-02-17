#!/bin/bash
#SBATCH --job-name=egemaps
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/egemaps_%j.out
#SBATCH --error=logs/egemaps_%j.err

cd $HOME/mci_detection
source .venv/bin/activate

echo "Running eGeMAPS extraction..."
python egemaps_extractor.py
echo "Done."
