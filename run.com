#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/hpc/41/dolamull/experiments/case-retrieval/output.log
#SBATCH --error=/storage/hpc/41/dolamull/experiments/case-retrieval/error.log

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /scratch/hpc/41/dolamull/conda_envs/llm_env
export HF_HOME=/scratch/hpc/41/dolamull/hf_cache

source <(grep -v '^#' .env | xargs -d '\n')

huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m preprocess.vectoriser --model_name $1 --dataset ilpcr
python -m preprocess.vectoriser --model_name $1 --dataset coliee
python -m preprocess.vectoriser --model_name $1 --dataset irled
python -m preprocess.vectoriser --model_name $1 --dataset muser
python -m preprocess.vectoriser --model_name $1 --dataset ecthr