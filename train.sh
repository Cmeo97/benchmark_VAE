#!/bin/bash

dataset=$1
model_name=$2
seed=$4
update_architecture=$3
beta=$5
alpha=$6
C=$7
latent_dim=$8


ExpName=${model_name}"_"${dataset}"_"${seed}"_"${update_architecture}
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



