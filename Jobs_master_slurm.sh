#!/bin/bash


ProjectName="Disentanlement_benchmark"



declare -a All_Datasets=(3Dshapes) 

declare -a All_Methods=(tc_vae)

declare -a All_seeds=(1)

declare -a All_architecture_updates=(False)

declare -a All_alphas=(0.15)

declare -a All_betas=(2)

declare -a All_Cs=(30)

declare -a All_latent_dims=(15)


for Dataset in "${All_Datasets[@]}"
do
	for Method in "${All_Methods[@]}"
	do
		for arc_update in "${All_architecture_updates[@]}"
		do	
			for seed in "${All_seeds[@]}"
			do
				for beta in "${All_betas[@]}"
				do
					for alpha in "${All_alphas[@]}"
					do
						for C in "${All_Cs[@]}"
						do
							for latent_dim in "${All_latent_dims[@]}"
							do
                			sbatch train_slurm.sh $Dataset $Method $arc_update $seed $beta $alpha $C $latent_dim 
							done
						done
					done
				done
            done
		done
	done
done

