#!/bin/bash
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --job-name=feas_tp_lora
#SBATCH --exclude=ai-gpu15
#SBATCH --output=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.err

set -e
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"
nvidia-smi -L || true

# Robust conda activation (same pattern as sanity sbatch - shared-apps profile +
# absolute env path). ai-gpu15 excluded because /home not mounted there.
CONDA_SH=$(ls /shared-apps/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-13.2.0/anaconda3-*/etc/profile.d/conda.sh 2>/dev/null | head -1)
if [ -z "$CONDA_SH" ]; then
  echo "ERROR: no shared-apps conda.sh found" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate /home/ahb4007/.conda/envs/nnunet
echo "CONDA_PREFIX=$CONDA_PREFIX"

# W&B
export WANDB_API_KEY="94ab84459aa1b42734e2980087f053e645c271e7"

cd /midtier/paetzollab/scratch/ahb4007/3D_segmentation

# Common CLI flags (paths remapped to /midtier for preempt)
COMMON_FLAGS="--config_path proposals/feasibility_textprompted_config.py \
  --fold 0 --max_samples 5 --epochs 200 --seed 42 \
  --text_prompted --batch_size 1 \
  --data_path /midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted \
  --precomputed_embeddings /midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt \
  --prompts_json proposals/textprompted_prompts_subset/ \
  --wandb_project 3D-Segmentation-sanity"

echo "RUN: LoRA r=16 + frozen encoder from VoxTell"
echo "extra flags: --init_checkpoint experiments/voxtell_converted/checkpoint.pth --freeze_encoder --lora --lora_rank 16"

python main.py $COMMON_FLAGS --init_checkpoint experiments/voxtell_converted/checkpoint.pth --freeze_encoder --lora --lora_rank 16

echo "Done at $(date)"
