#!/bin/bash
#SBATCH -p gpu
#SBATCH -o logs/slurm_gpu.out
#SBATCH -e logs/slurm_gpu.err


# Fallback if someone runs this with srun/bash directly (not sbatch)
# if [ -z "$SLURM_JOB_ID" ]; then
#     echo "No Slurm allocation detected — requesting one with srun..."
#     exec srun -p gpu "$0" "$@"
# fi


module load qibo
source ~/envs/qibo_env/bin/activate


export CUDA_VISIBLE_DEVICES=0

python3 scripts/scripts_executor.py --device numpy
