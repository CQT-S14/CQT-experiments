#!/bin/bash
#SBATCH -p sinq20
#SBATCH -o logs/slurm_sinq20_dev_%j.out
#SBATCH -e logs/slurm_sinq20_dev_%j.err

echo "Log files: logs/slurm_sinq20_dev_${SLURM_JOB_ID}.out / .err"


# Resolve packages and dependencies in this order:
# First: package versions specified in dev_env as defined in setup_dev_env.sh 
# Second: falls back to module qibo packages if packages not  present in dev_env (eg. keysight)
# 
# Esesentially: editable installs take priority over module ones which are put
# into zzz_module_fallback.pth automatically so they are forced to resolve only if needed.
module load qibo
unset PYTHONPATH                    # unset path set by module qibo

# Locate the repo root regardless of where sbatch was called from.
# Under Slurm, scontrol gives us the original script path (works from any cwd).
# Fallback to BASH_SOURCE for direct execution (e.g. bash pipeline/run_sinq20_dev.sh).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP '(?<=Command=)\S+')
    REPO_ROOT=$(cd "$(dirname "$_SCRIPT")/.." && pwd)
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi

# Source setup_dev_env.sh to get ENV_DIR and REPOS (no setup runs, only variables)
source "$REPO_ROOT/workspace/setup_dev_env.sh"

source "$ENV_DIR/bin/activate"      # get packages from dev_env

export CUDA_VISIBLE_DEVICES=0

# Log package versions at the start of every run for traceability
echo "===== DEPENDENCY VERSIONS ====="
for pkg in qibocal qibolab qibo; do
  pip show "$pkg" | grep -E "^(Name|Version|Location)"
  loc=$(pip show "$pkg" 2>/dev/null | grep "^Editable project location:" | cut -d' ' -f4)
  if [ -n "$loc" ] && [ -d "$loc/.git" ]; then
    branch=$(git -C "$loc" branch --show-current)
    commit=$(git -C "$loc" rev-parse --short HEAD)
    echo "Git: branch=$branch commit=$commit (editable: $loc)"
  fi
  echo "---"
done
echo "==============================="

cd ~/"${REPOS[0]}"
python3 scripts/scripts_executor.py --device sinq20
