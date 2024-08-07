#!/bin/bash

#SBATCH --job-name=dscal
#SBATCH --gpus=1
#SBATCH --time=00:01:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd
#SBATCH --account=ta163

# Set modules
module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan

# Check assigned GPU
srun --ntasks=1 rocm-smi

srun --ntasks=1 --cpus-per-task=1 ./a.out

exit 0
