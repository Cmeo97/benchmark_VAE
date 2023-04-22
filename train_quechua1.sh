#!/bin/bash

dataset=$1
model_name=$2
latent_dim=$3
seed=$4
beta=$5
alpha=$6
C=$7



ExpName=${model_name}"_"${dataset}"_"${seed}"_"${beta}"_"${alpha}"_"${C}"_"${latent_dim}
echo "Training of experiment: ${ExpName}"

python examples/scripts/training.py \
--dataset=${dataset} \
--model_name=${model_name} \
--model_config=/users/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/${model_name}_config.json \
--training_config=/users/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/base_training_config.json \
--use_comet \
--seed=${seed} \
--beta=${beta} \
--C=${C} \
--alpha=${alpha} \
--latent_dim=${latent_dim} \
--name_exp=${ExpName} \
--data_path=$DATA_PATH \
#> logs/${ExpName}.out 2> logs/${ExpName}.err 
#echo "Evaluation of experiment: ${ExpName}"
#python examples/scripts/validation.py \
#--dataset=${dataset} \
#--model_name=${model_name} \
#--model_config=/home/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/${model_name}_config.json \
#--training_config=/home/cristianmeo/benchmark_VAE/examples/scripts/configs/${dataset}/base_training_config.json \
#--exp_name=${ExpName} \
#--data_path=$DATA_PATH \
#--enc_celeba=${enc_celeba} \
#--dec_celeba=${dec_celeba} \



