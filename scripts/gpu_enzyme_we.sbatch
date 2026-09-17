#!/bin/bash
#SBATCH --job-name=weqmmm-70atom-we
#SBATCH --account=general
#SBATCH --partition=general-short
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=validation/gpu-70atom-we-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate_gpu.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Pass the completed 70-atom MD directory as the first script argument.
# Submit only with afterok dependencies on MD and GPU NVE/runner gates.
pilot_directory="${1:?Pass the 70-atom pilot MD directory}"
python examples/hca2_azm/run_we.py --config examples/hca2_azm/def2svp_gpu.yaml --state "$pilot_directory/state-000010.npz" --metric "$pilot_directory/pilot-metric.json" --output "runs/hca2_azm/def2svp-gpu-we-${SLURM_JOB_ID}" --walkers 4 --workers 2 --cycles 3 --steps 2
