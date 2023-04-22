#!/bin/bash

#SBATCH --account=research-eemcs-me
#SBATCH --job-name=tc_vae
#SBATCH --partition=gpu                       
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu
#SBATCH --mem=120G     
#SBATCH --ntasks=1                                
#SBATCH --time=23:59:00
#SBATCH -o /scratch/cristianmeo/output/tc-vae-%A_%a.out  
#SBATCH -e /scratch/cristianmeo/output/tc-vae-%A_%a.err  

module --quiet load miniconda3/4.12.0
conda activate MARL

dataset=$1
model_name=$2
latent_dim=$3
seed=$4
beta=$5
alpha=$6
C=$7
enc_celeba=$8
dec_celeba=$9


ExpName=${model_name}"_"${dataset}"_"${seed}"_"${beta}"_"${alpha}"_"${C}"_"${latent_dim}"_"${enc_celeba}"_"${dec_celeba}
echo "Training of experiment: ${ExpName}"

python examples/scripts/training.py \
--dataset=${dataset} \
--model_name=${model_name} \
--model_config=/home/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/${model_name}_config.json \
--training_config=/home/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/base_training_config.json \
--use_comet \
--seed=${seed} \
--beta=${beta} \
--C=${C} \
--alpha=${alpha} \
--latent_dim=${latent_dim} \
--name_exp=${ExpName} \
--data_path=$DATA_PATH \
--use_hpc \
--enc_celeba=${enc_celeba} \
--dec_celeba=${dec_celeba} \
> logs/${ExpName}.out 2> logs/${ExpName}.err 
echo "Evaluation of experiment: ${ExpName}"
python examples/scripts/validation.py \
--dataset=${dataset} \
--model_name=${model_name} \
--model_config=/home/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/${model_name}_config.json \
--training_config=/home/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/base_training_config.json \
--exp_name=${ExpName} \
--data_path=$DATA_PATH \
--enc_celeba=${enc_celeba} \
--dec_celeba=${dec_celeba} \
--use_hpc \


