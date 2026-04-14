#!/bin/bash
#SBATCH --job-name=feas_tp_a
#SBATCH --output=/ministorage/ahb/3D_segmentation/slurm_logs/%x_%j.out
#SBATCH --error=/ministorage/ahb/3D_segmentation/slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=minilab-gpu
#SBATCH --qos=normal

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

module load anaconda3 2>/dev/null || true
source $CONDA_PREFIX/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate nnunet

export WANDB_API_KEY="94ab84459aa1b42734e2980087f053e645c271e7"

cd /ministorage/ahb/3D_segmentation

nvidia-smi
echo ""

python main.py \
    --config_path proposals/feasibility_textprompted_config.py \
    --fold 0 \
    --max_samples 1 \
    --batch_size 2 \
    --epochs 100 \
    --seed 42 \
    --text_prompted \
    --precomputed_embeddings /ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt \
    --prompts_json proposals/textprompted_prompts_subset/ \
    --wandb_project 3D-Segmentation-sanity

EXIT=$?
echo "End: $(date), exit=$EXIT"
exit $EXIT
