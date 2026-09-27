#!/bin/bash
# Script to automate setup on Ada HPC (ada.iiit.ac.in) for user bhogadi.likhit

set -e

echo "=== Setting up Amazon ML GPU Environment on Ada HPC ==="

ENV_NAME="amazon_gpu"

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Allocating compute node on 'u22' partition with 16GB RAM for setup..."
    srun --partition=u22 --mem=16G bash -c "
        set -e
        eval \"\$(\$HOME/miniconda/bin/conda shell.bash hook)\"
        
        if ! conda info --envs | grep -q \"$ENV_NAME\"; then
            echo 'Creating Conda environment \"$ENV_NAME\" with Python 3.10...'
            conda create -n $ENV_NAME python=3.10 -y
        fi

        source \$(\$HOME/miniconda/bin/conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME

        echo 'Installing PyTorch with CUDA support...'
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        echo 'Installing project dependencies...'
        pip install -r ~/amazonMl/requirements.txt

        echo '=== Verification ==='
        python ~/amazonMl/check_gpu.py

        echo '=== Setup Complete! ==='
    "
else
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    if ! conda info --envs | grep -q "$ENV_NAME"; then
        conda create -n $ENV_NAME python=3.10 -y
    fi
    source $($HOME/miniconda/bin/conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r ~/amazonMl/requirements.txt
    python ~/amazonMl/check_gpu.py
fi
