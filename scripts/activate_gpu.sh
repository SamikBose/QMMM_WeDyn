# Source from bash on an allocated GPU node. Uses the tested CPU environment's
# base interpreter/dependencies, with pinned GPU dependencies in a local venv.
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
module load CUDA/12.4.0
source "$WEQMMM_ROOT/.venv-gpu/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
# Use only the device assigned by Slurm; do not replace CUDA_VISIBLE_DEVICES.
