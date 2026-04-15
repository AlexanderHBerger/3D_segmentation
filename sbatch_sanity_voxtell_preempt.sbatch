#!/bin/bash
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=sanity_voxtell
#SBATCH --exclude=ai-gpu15
#SBATCH --output=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.err

set -e
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"

# Robust conda activation: source shared-apps profile, then activate by
# absolute env path. Previous `conda activate nnunet` call failed on
# ai-gpu15 where /home is not mounted (NFS-mounted on other preempt
# nodes). `--exclude=ai-gpu15` above keeps the job on nodes with /home.
CONDA_SH=$(ls /shared-apps/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-13.2.0/anaconda3-*/etc/profile.d/conda.sh 2>/dev/null | head -1)
if [ -z "$CONDA_SH" ]; then
  echo "ERROR: no shared-apps conda.sh found" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate /home/ahb4007/.conda/envs/nnunet
echo "CONDA_PREFIX=$CONDA_PREFIX"
python -c "import sys; print('python=' + sys.executable)"

cd /midtier/paetzollab/scratch/ahb4007/3D_segmentation

# (1) Standard text-prompted sanity check (must pass before any further work)
python scripts/sanity_check.py --text-prompted

# (2) VoxTell-init preflight (state_dict load, param counts, grad flow)
CHECKPOINT_PATH=/midtier/paetzollab/scratch/ahb4007/3D_segmentation/experiments/voxtell_converted/checkpoint.pth \
    python proposals/voxtell_preflight.py

echo "Done at $(date)"
