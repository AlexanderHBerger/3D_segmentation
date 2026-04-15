#!/bin/bash
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=sanity_voxtell
#SBATCH --output=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.err

set -e
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"

eval "$(conda shell.bash hook)"
conda activate nnunet

cd /midtier/paetzollab/scratch/ahb4007/3D_segmentation

# (1) Standard text-prompted sanity check (must pass before any further work)
python scripts/sanity_check.py --text-prompted

# (2) VoxTell-init preflight (state_dict load, param counts, grad flow)
CHECKPOINT_PATH=/midtier/paetzollab/scratch/ahb4007/3D_segmentation/experiments/voxtell_converted/checkpoint.pth \
    python proposals/voxtell_preflight.py

echo "Done at $(date)"
