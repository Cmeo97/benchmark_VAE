#!/bin/bash

#SBATCH --account=research-eemcs-me
#SBATCH --job-name=tc_vae
#SBATCH --partition=gpu                       
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu
#SBATCH --mem=120G     
#SBATCH --ntasks=1                                
#SBATCH --time=2:25:00
#SBATCH -o /scratch/cristianmeo/output/tc-vae-%A.out  
#SBATCH -e /scratch/cristianmeo/output/tc-vae-%A.err  

module --quiet load miniconda3/4.12.0
conda activate MARL

dataset=$1
model_name=$2
latent_dim=$3
seed=$4
beta=$5
alpha=$6
C=$7
enc_celeba=True
dec_celeba=False


ExpName=${model_name}"_"${dataset}"_"${seed}"_"${beta}"_"${alpha}"_"${C}"_"${latent_dim}"_"${enc_celeba}"_"${dec_celeba}
echo "validating experiment: ${ExpName}"

python downstream_task_validation.py \
--dataset=${dataset} \
--model_name=${model_name} \
--exp_name=${ExpName} \
--data_path=$DATA_PATH \
--latent_dim=${latent_dim} \
#> logs/${ExpName}"_downstream_task".out 2> logs/${ExpName}"_downstream_task".err 

