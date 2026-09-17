#!/bin/bash
#SBATCH --job-name=weqmmm-70atom
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=validation/gpu-70atom-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate_gpu.sh
python scripts/run_md.py --config examples/hca2_azm/def2svp_gpu.yaml --steps 0 --output "runs/hca2_azm/def2svp-gpu-single-point-${SLURM_JOB_ID}"
