#!/bin/bash


ProjectName="Disentanlement_benchmark"


declare -a All_Datasets=(3Dshapes dsprites) #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"

declare -a All_Methods=(disentangled_beta_vae tc_vae)

declare -a All_seeds=(1 2 3 4 5)

declare -a All_architecture_updates=(False True)

for Dataset in "${All_Datasets[@]}"
do
	for Method in "${All_Methods[@]}"
	do
		for arc_update in "${All_architecture_updates[@]}"
		do	
			for seed in "${All_seeds[@]}"
			do
                bash train.sh $Dataset $Method $arc_update $seed &
            done
		done
	done
done
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for grid_size in "${All_grid_sizes[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/marlgrid/OOD_tests/OOD_size_env_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $grid_size		
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done