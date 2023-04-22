#!/bin/bash

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

python examples/scripts/validation.py \
--dataset=${dataset} \
--model_name=${model_name} \
--training_config=/users/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/base_training_config.json \
--exp_name=${ExpName} \
--data_path=$DATA_PATH \
#> logs/${ExpName}"_validation".out 2> logs/${ExpName}"_validation".err 

