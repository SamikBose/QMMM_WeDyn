#!/bin/bash
#SBATCH --job-name=weqmmm-70atom-md
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=validation/gpu-70atom-md-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate_gpu.sh
# Submit with afterok dependency on the full 70-atom single-point job.
python scripts/run_md.py --config examples/hca2_azm/def2svp_gpu.yaml --steps 10 --report-interval 1 --save-qm-result --output "runs/hca2_azm/def2svp-gpu-md-${SLURM_JOB_ID}"
python scripts/build_metric.py --pilot-dir "runs/hca2_azm/def2svp-gpu-md-${SLURM_JOB_ID}" --output "runs/hca2_azm/def2svp-gpu-md-${SLURM_JOB_ID}/pilot-metric.json"
