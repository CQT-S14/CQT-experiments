#!/bin/bash
#SBATCH -p sinq20
#SBATCH -o logs/slurm_sinq20.out
#SBATCH -e logs/slurm_sinq20.err

# Fallback if someone runs this with srun/bash directly (not sbatch)
# if [ -z "$SLURM_JOB_ID" ]; then
#     echo "No Slurm allocation detected — requesting one with srun..."
#     exec srun -p sinq20 "$0" "$@"
# fi

module load qibo
source ~/envs/qibo_env/bin/activate

export CUDA_VISIBLE_DEVICES=0

# Change to project directory to ensure relative paths work correctly
cd ~/CQT-experiments

python3 scripts/scripts_executor.py --device sinq20

