#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/hpc/41/dolamull/experiments/case-retrieval/output.log
#SBATCH --error=/storage/hpc/41/dolamull/experiments/case-retrieval/error.log

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /scratch/hpc/41/dolamull/conda_envs/llm_env_clone
export HF_HOME=/scratch/hpc/41/dolamull/hf_cache

source <(grep -v '^#' .env | xargs -d '\n')

huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m preprocess.vectoriser --model_name BAAI/bge-en-icl --dataset lecardv2
python -m preprocess.vectoriser --model_name Salesforce/SFR-Embedding-2_R --dataset lecardv2