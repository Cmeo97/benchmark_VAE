#!/bin/bash

dataset=$1
model_name=$2
latent_dim=$3
seed=$4
beta=$5
alpha=$6
C=$7


ExpName=${model_name}"_"${dataset}"_"${seed}"_"${beta}"_"${alpha}"_"${C}"_"${latent_dim}
echo "doing experiment: ${ExpName}"

python examples/scripts/validation.py \
--dataset=${dataset} \
--model_name=${model_name} \
--model_config=/users/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/${model_name}_config.json \
--training_config=/users/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/base_training_config.json \
--exp_name=${ExpName} \
--data_path=$DATA_PATH \

