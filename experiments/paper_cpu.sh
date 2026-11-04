#!/bin/bash
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
poetry run python "$@"
