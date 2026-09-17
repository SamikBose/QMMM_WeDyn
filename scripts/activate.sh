# Source this file from bash: source scripts/activate.sh
# Keep PartQMMM and the project importable without altering the user's global Python.
WEQMMM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate weqmmm
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WEQMMM_ROOT:$WEQMMM_ROOT/external/PartQMMM${PYTHONPATH:+:$PYTHONPATH}"
