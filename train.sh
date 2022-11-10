#!/bin/bash

dataset=$1
model_name=$2
update_architecture=$3
seed=$4
beta=$5
alpha=$6
C=$7
latent_dim=$8


ExpName=${model_name}"_"${dataset}"_"${seed}"_"${update_architecture}"_"${beta}"_"${alpha}"_"${C}"_"${latent_dim}
echo "doing experiment: ${ExpName}"

nohup python examples/scripts/training.py \
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
> logs/${ExpName}.out 2> logs/${ExpName}.err &


