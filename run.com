#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk
#SBATCH --output=/storage/hpc/41/dolamull/experiments/case-retrieval/output.log
#SBATCH --error=/storage/hpc/41/dolamull/experiments/case-retrieval/error.log

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/41/dolamull/conda_envs/llm_env
export HF_HOME=/scratch/hpc/41/dolamull/hf_cache

source <(grep -v '^#' .env | xargs -d '\n')

huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m preprocess.vectoriser --model_name nvidia/NV-Embed-v2 --dataset IL_PCR