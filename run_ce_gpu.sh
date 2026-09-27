#!/bin/bash
#SBATCH --job-name=crossencoder_train
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ce_%j.out
#SBATCH --error=logs/ce_%j.err

# Load bashrc and activate conda
source ~/.bashrc
eval "$($HOME/miniconda/bin/conda shell.bash hook 2>/dev/null || true)"
conda activate amazon_gpu 2>/dev/null || source activate amazon_gpu

mkdir -p logs

echo "=========================================================="
echo "Starting Cross-Encoder Training & Inference on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================================="

nvidia-smi

cd ~/amazonMl/code/business_entity_resolution/src
export ER_CE_DIR=~/amazonMl/work/ce

# 1. Train Cross-Encoder model on hidden set pairs H
echo "--- Step 1: Training Cross-Encoder model ---"
python crossencoder.py train --model intfloat/multilingual-e5-small --out ~/amazonMl/work/ce/ce_model --epochs 2 --bs 128 --lr 5e-5

# 2. Score candidate bands for train and test sets
echo "--- Step 2: Scoring train band ---"
python crossencoder.py infer --model ~/amazonMl/work/ce/ce_model --band band_train.parquet --out ~/amazonMl/work/ce/ce_train.parquet --bs 512

echo "--- Step 3: Scoring test band ---"
python crossencoder.py infer --model ~/amazonMl/work/ce/ce_model --band band_test.parquet --out ~/amazonMl/work/ce/ce_test.parquet --bs 512

echo "=========================================================="
echo "Cross-Encoder Completed at: $(date)"
echo "=========================================================="
