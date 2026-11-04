#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=24:00:00
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
poetry run python "$@"
