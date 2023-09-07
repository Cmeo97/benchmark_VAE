#!/bin/bash

ProjectName="Disentanlement_benchmark"
## celeba


#declare -a All_Datasets=(3Dshapes) 
##
#declare -a All_Methods=(beta_tc_vae)
##
#declare -a All_seeds=(0 1)
##
#declare -a All_alphas=(0)
##
#declare -a All_betas=(3)
##
#declare -a All_Cs=(30)
##
#declare -a All_latent_dims=(10)
##
#declare -a All_decoder=(False)
##
#declare -a All_encoder=(True)
##
#declare -a All_imbalance=(3)
##
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C $imbalance
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#declare -a All_Datasets=(teapots) 
#
#declare -a All_Methods=(beta_tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0)
#
#declare -a All_betas=(2)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#
declare -a All_Datasets=(celeba) 

declare -a All_Methods=(tc_vae)

declare -a All_seeds=(0 1 2 3 4) 

declare -a All_alphas=(0.25 0.5 0.75)

declare -a All_betas=(5)

declare -a All_Cs=(50)

declare -a All_latent_dims=(48)

declare -a All_decoder=(False)

declare -a All_encoder=(True)

declare -a All_imbalance=(0)

for Dataset in "${All_Datasets[@]}"
do
	for Method in "${All_Methods[@]}"
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
							for decoder in "${All_decoder[@]}"
							do
								for encoder in "${All_encoder[@]}"
								do
									for imbalance in "${All_imbalance[@]}"
									do
                						sbatch train_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C  
									done
								done
							done 
						done
					done
				done
            done
		done
	done
done

#declare -a All_Datasets=(3Dshapes) 
#
#declare -a All_Methods=(tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0.25 0.5 0.75)
#
#declare -a All_betas=(1 3)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                						sbatch train_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C  
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#

#declare -a All_Datasets=(teapots) 
#
#declare -a All_Methods=(tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0.25 0.5 0.75)
#
#declare -a All_betas=(1 2)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                						sbatch train_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C  
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#
#
##
##declare -a All_Datasets=(celeba) 
##
##declare -a All_Methods=(tc_vae)
##
##declare -a All_seeds=(1 2)
##
##declare -a All_alphas=(0)
##
##declare -a All_betas=(5)
##
##declare -a All_Cs=(50)
##
##declare -a All_latent_dims=(32)
##
##declare -a All_decoder=(False)
##
##declare -a All_encoder=(True)
##
##declare -a All_imbalance=(0)
##
##for Dataset in "${All_Datasets[@]}"
##do
##	for Method in "${All_Methods[@]}"
##	do
##		for seed in "${All_seeds[@]}"
##		do
##			for beta in "${All_betas[@]}"
##			do
##				for alpha in "${All_alphas[@]}"
##				do
##					for C in "${All_Cs[@]}"
##					do
##						for latent_dim in "${All_latent_dims[@]}"
##						do
##							for decoder in "${All_decoder[@]}"
##							do
##								for encoder in "${All_encoder[@]}"
##								do
##									for imbalance in "${All_imbalance[@]}"
##									do
##                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
##									done
##								done
##							done 
##						done
##					done
##				done
##            done
##		done
#	done
#done

#declare -a All_Datasets=(celeba) 
#
#declare -a All_Methods=(tc_vae)
#
#declare -a All_seeds=(1 2 3)
#
#declare -a All_alphas=(0.25 0.75)
#
#declare -a All_betas=(5)
#
#declare -a All_Cs=(50)
#
#declare -a All_latent_dims=(16)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#



#
#declare -a All_Datasets=(cifar10) 
#
#declare -a All_Methods=(tc_vae)
#
#declare -a All_seeds=(2 3)
#
#declare -a All_alphas=(0.25 0.75)
#
#declare -a All_betas=(2)
#
#declare -a All_Cs=(100)
#
#declare -a All_latent_dims=(64)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch train_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#
#declare -a All_Datasets=(cifar10) 
#
#declare -a All_Methods=(disentangled_beta_vae factor_vae)
#
#declare -a All_seeds=(2 3)
#
#declare -a All_alphas=(0)
#
#declare -a All_betas=(2)
#
#declare -a All_Cs=(100)
#
#declare -a All_latent_dims=(64)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch train_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#
#declare -a All_Datasets=(3Dshapes) 
#
#declare -a All_Methods=(beta_tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0)
#
#declare -a All_betas=(3)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
##declare -a All_Datasets=(3Dshapes) 
##
##declare -a All_Methods=(tc_vae)
##
##declare -a All_seeds=(1 2 3)
##
##declare -a All_alphas=(0.25 0.75)
##
##declare -a All_betas=(3)
##
##declare -a All_Cs=(30)
##
##declare -a All_latent_dims=(10)
##
##declare -a All_decoder=(False)
##
##declare -a All_encoder=(True)
##
##declare -a All_imbalance=(0)
##
##for Dataset in "${All_Datasets[@]}"
##do
##	for Method in "${All_Methods[@]}"
##	do
##		for seed in "${All_seeds[@]}"
##		do
##			for beta in "${All_betas[@]}"
##			do
##				for alpha in "${All_alphas[@]}"
##				do
##					for C in "${All_Cs[@]}"
##					do
##						for latent_dim in "${All_latent_dims[@]}"
##						do
##							for decoder in "${All_decoder[@]}"
##							do
##								for encoder in "${All_encoder[@]}"
##								do
##									for imbalance in "${All_imbalance[@]}"
##									do
##                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
##									done
##								done
##							done 
##						done
##					done
##				done
##            done
##		done
##	done
##done
#
#declare -a All_Datasets=(teapots) 
#
#declare -a All_Methods=(beta_tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0)
#
#declare -a All_betas=(2)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
##declare -a All_Datasets=(teapots) 
##
##declare -a All_Methods=(tc_vae)
##
##declare -a All_seeds=(1 2 3)
##
##declare -a All_alphas=(0.25 0.75)
##
##declare -a All_betas=(2)
##
##declare -a All_Cs=(30)
##
##declare -a All_latent_dims=(10)
##
##declare -a All_decoder=(False)
##
##declare -a All_encoder=(True)
##
##declare -a All_imbalance=(0)
##
##for Dataset in "${All_Datasets[@]}"
##do
##	for Method in "${All_Methods[@]}"
##	do
##		for seed in "${All_seeds[@]}"
##		do
##			for beta in "${All_betas[@]}"
##			do
##				for alpha in "${All_alphas[@]}"
##				do
##					for C in "${All_Cs[@]}"
##					do
##						for latent_dim in "${All_latent_dims[@]}"
##						do
##							for decoder in "${All_decoder[@]}"
##							do
##								for encoder in "${All_encoder[@]}"
##								do
##									for imbalance in "${All_imbalance[@]}"
##									do
##                					sbatch validation_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C 
##									done
##								done
##							done 
##						done
##					done
##				done
##            done
##		done
##	done
##done
##
#### with imbalance 
#
#
#declare -a All_Datasets=(3Dshapes) 
#
#declare -a All_Methods=(beta_tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0)
#
#declare -a All_betas=(3)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0.25 0.50 1.00)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_imbalance_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C $imbalance
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
#
#
#declare -a All_Datasets=(teapots) 
#
#declare -a All_Methods=(beta_tc_vae)
#
#declare -a All_seeds=(0 1)
#
#declare -a All_alphas=(0)
#
#declare -a All_betas=(2)
#
#declare -a All_Cs=(30)
#
#declare -a All_latent_dims=(10)
#
#declare -a All_decoder=(False)
#
#declare -a All_encoder=(True)
#
#declare -a All_imbalance=(0.25 0.50 1.00)
#
#for Dataset in "${All_Datasets[@]}"
#do
#	for Method in "${All_Methods[@]}"
#	do
#		for seed in "${All_seeds[@]}"
#		do
#			for beta in "${All_betas[@]}"
#			do
#				for alpha in "${All_alphas[@]}"
#				do
#					for C in "${All_Cs[@]}"
#					do
#						for latent_dim in "${All_latent_dims[@]}"
#						do
#							for decoder in "${All_decoder[@]}"
#							do
#								for encoder in "${All_encoder[@]}"
#								do
#									for imbalance in "${All_imbalance[@]}"
#									do
#                					sbatch validation_imbalance_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C $imbalance
#									done
#								done
#							done 
#						done
#					done
#				done
#            done
#		done
#	done
#done
#
##declare -a All_Datasets=(teapots) 
##
##declare -a All_Methods=(tc_vae)
##
##declare -a All_seeds=(0 1)
##
##declare -a All_alphas=(0.25 0.75)
##
##declare -a All_betas=(2)
##
##declare -a All_Cs=(30)
##
##declare -a All_latent_dims=(10)
##
##declare -a All_decoder=(False)
##
##declare -a All_encoder=(True)
##
##declare -a All_imbalance=(0.25 0.50 1.00)
##
##for Dataset in "${All_Datasets[@]}"
##do
##	for Method in "${All_Methods[@]}"
##	do
##		for seed in "${All_seeds[@]}"
##		do
##			for beta in "${All_betas[@]}"
##			do
##				for alpha in "${All_alphas[@]}"
##				do
##					for C in "${All_Cs[@]}"
##					do
##						for latent_dim in "${All_latent_dims[@]}"
##						do
##							for decoder in "${All_decoder[@]}"
##							do
##								for encoder in "${All_encoder[@]}"
##								do
##									for imbalance in "${All_imbalance[@]}"
##									do
##                					sbatch train_imbalance_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C $imbalance
##									done
##								done
##							done 
##						done
##					done
##				done
##            done
##		done
##	done
##done
##
##declare -a All_Datasets=(teapots) 
##
##declare -a All_Methods=(factor_vae disentangled_beta_vae)
##
##declare -a All_seeds=(0 1)
##
##declare -a All_alphas=(0)
##
##declare -a All_betas=(2)
##
##declare -a All_Cs=(30)
##
##declare -a All_latent_dims=(10)
##
##declare -a All_decoder=(False)
##
##declare -a All_encoder=(True)
##
##declare -a All_imbalance=(0.25 0.50 1.00)
##
##for Dataset in "${All_Datasets[@]}"
##do
##	for Method in "${All_Methods[@]}"
##	do
##		for seed in "${All_seeds[@]}"
##		do
##			for beta in "${All_betas[@]}"
##			do
##				for alpha in "${All_alphas[@]}"
##				do
##					for C in "${All_Cs[@]}"
##					do
##						for latent_dim in "${All_latent_dims[@]}"
##						do
##							for decoder in "${All_decoder[@]}"
##							do
##								for encoder in "${All_encoder[@]}"
##								do
##									for imbalance in "${All_imbalance[@]}"
##									do
##                					sbatch train_imbalance_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C $imbalance
##									done
##								done
##							done 
##						done
##					done
##				done
##            done
##		done
##	done
##done
##
###declare -a All_Datasets=(3Dshapes) 
###
###declare -a All_Methods=(beta_tc_vae)
###
###declare -a All_seeds=(1 2 3)
###
###declare -a All_alphas=(0.25 0.75)
###
###declare -a All_betas=(3)
###
###declare -a All_Cs=(30)
###
###declare -a All_latent_dims=(10)
###
###declare -a All_decoder=(False)
###
###declare -a All_encoder=(True)
###
###declare -a All_imbalance=(0.25 0.50 1.00)
###
###for Dataset in "${All_Datasets[@]}"
###do
###	for Method in "${All_Methods[@]}"
###	do
###		for seed in "${All_seeds[@]}"
###		do
###			for beta in "${All_betas[@]}"
###			do
###				for alpha in "${All_alphas[@]}"
###				do
###					for C in "${All_Cs[@]}"
###					do
###						for latent_dim in "${All_latent_dims[@]}"
###						do
###							for decoder in "${All_decoder[@]}"
###							do
###								for encoder in "${All_encoder[@]}"
###								do
##									for imbalance in "${All_imbalance[@]}"
##									do
##                					sbatch validation_imbalance_slurm.sh $Dataset $Method $latent_dim $seed $beta $alpha $C $imbalance
##									done
##								done
##							done 
##						done
##					done
##				done
##            done
##		done
##	done
##done
##