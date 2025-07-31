#!/bin/bash

# Exit immediately on error
set -e

# Create the environment
echo "Creating Conda environment 'gensvs_env'..."
conda env create -f "./env_info/gensvs_env.yml"

# Activate the environment
echo "Activating environment..."
# NOTE: This only works in interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gensvs_env

# Set CUDA_HOME environment variable
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
echo "CUDA_HOME set to: $CUDA_HOME"

# install gensvs package
echo "Installing gensvs package..."
pip install -e .