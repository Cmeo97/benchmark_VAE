#!/bin/bash

python training.py --dataset celeba --model_name ae --model_config 'configs/celeba/ae_config.json' --training_config 'configs/celeba/base_training_config.json'
python training.py --dataset celeba --model_name vae --model_config 'configs/celeba/vae_config.json' --training_config 'configs/celeba/base_training_config.json'
python training.py --dataset celeba --model_name wae --model_config 'configs/celeba/wae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name vamp --model_config 'configs/celeba/vamp_config.json' --training_config 'configs/celeba/base_training_config.json'
python training.py --dataset celeba --model_name rae_l2 --model_config 'configs/celeba/rae_l2_config.json' --training_config 'configs/celeba/base_training_config.json'
python training.py --dataset celeba --model_name rae_gp --model_config 'configs/celeba/rae_gp_config.json' --training_config 'configs/celeba/base_training_config.json'
python training.py --dataset celeba --model_name hvae --model_config 'configs/celeba/hvae_config.json' --training_config 'configs/celeba/base_training_config.json'
python training.py --dataset celeba --model_name rhvae --model_config 'configs/celeba/rhvae_config.json' --training_config 'configs/celeba/base_training_config.json'
