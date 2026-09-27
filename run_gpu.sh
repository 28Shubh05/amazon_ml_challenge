#!/bin/bash
#SBATCH --job-name=amazon_ml_gpu
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Load bashrc and activate conda
source ~/.bashrc
eval "$($HOME/miniconda/bin/conda shell.bash hook 2>/dev/null || true)"
conda activate amazon_gpu 2>/dev/null || source activate amazon_gpu

# Ensure log directory exists
mkdir -p logs

echo "=========================================================="
echo "Job Started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================================="

# Check GPU status
nvidia-smi

# Verify GPU PyTorch environment
python check_gpu.py

# Run main pipeline / training script
# python code/business_entity_resolution/src/pipeline.py

echo "=========================================================="
echo "Job Finished at: $(date)"
echo "=========================================================="
