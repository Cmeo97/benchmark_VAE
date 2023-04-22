import argparse
import logging
import os
import numpy as np
import torch
from pythae.trainers import (
    BaseTrainerConfig,
    AdversarialTrainerConfig,
)
from pythae.models import AutoModel
from pythae.pipelines.metrics import EvaluationPipeline
from torchvision import transforms, datasets
from pythae.data.datasets import Dataset
import json
from pythae.data.shapes3d import Shapes3D
from pythae.data.teapots import Teapots
from pythae.data.celeba import Celeba

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cifar10", "celeba","dsprites", "3Dshapes", "teapots"],
    help="The data set to use to perform training. It must be located in the folder 'data' at the "
    "path 'data/datset_name/' and contain a 'train_data.npz' and a 'eval_data.npz' file with the "
    "data being under the key 'data'. The data must be in the range [0-255] and shaped with the "
    "channel in first position (im_channel x height x width).",
    required=True,
)
ap.add_argument(
    "--model_name",
    help="The name of the model to train",
    choices=[
        "disentangled_beta_vae",
        "factor_vae",
        "beta_tc_vae",
        "tc_vae",
    ],
    required=True,
)
ap.add_argument(
    "--exp_name", 
    type=str,
    help='name experiment',
    default='None',
)
ap.add_argument(
    "--training_config",
    help="path to training config_file (expected json file)",
    default=os.path.join(PATH, "configs/base_training_config.json"),
)
ap.add_argument(
    "--data_path",
    help='dataset folder path ',
    type=str,
    default="/home/cristianmeo/Datasets/",
)
args = ap.parse_args()


def main(args):
    
    if args.dataset == "3Dshapes": 
        Shapes3D_PATH='/home/cristianmeo/Datasets/3dshapes.h5'
        data = Shapes3D(Shapes3D_PATH)
        data_n_split = int(data.images.shape[0]*0.8)
        #train_data = data.images[:data_n_split]
        eval_data = data.images[data_n_split:]
        #train_labels = data.labels[:data_n_split]
        eval_labels = data.labels[data_n_split:]
       
    if args.dataset == "celeba":
      
        # Spatial size of training images, images are resized to this size.
        image_size = 64
        Celeba_PATH=args.data_path+'celeba/img_align_celeba'
        # Transformations to be applied to each individual image sample
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
        # Load the dataset from file and apply transformations
        data = Celeba(Celeba_PATH, transform)
        data_n_split = int(data.images.shape[0]*0.8)
        #train_data = data.images[:data_n_split]
        eval_data = data.images[data_n_split:]
        #train_labels = data.labels[:data_n_split]
        eval_labels = data.labels[data_n_split:]

    if args.dataset == "teapots":

        Teapots_PATH='/home/cristianmeo/Datasets/teapots/teapots.npz'
        data = Teapots(Teapots_PATH)
        data_n_split = int(data.images.shape[0]*0.8)
        #train_data = data.images[:data_n_split]
        eval_data = data.images[data_n_split:]
        #train_labels = data.labels[:data_n_split]
        eval_labels = data.labels[data_n_split:]

    if args.dataset == "cifar10":
   
        image_size=64
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(image_size)])

        eval_data = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform).data.transpose((0, 3, 1, 2))/255


    eval_dataset = Dataset(eval_data, eval_labels)
 

    if args.model_name == "FactorVAE":
        training_config = AdversarialTrainerConfig.from_json_file(args.training_config)

    else:
        training_config = BaseTrainerConfig.from_json_file(args.training_config)

    logger.info(f"Training config: {training_config}\n")



      
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    freer_device = np.argmin(memory_available)
    device = (
        "cuda:1"  # freer_device if device needs to be set, only "cuda" if using hpc
        if torch.cuda.is_available() and not training_config.no_cuda
        else "cpu"
    )

    model_path = '/users/cristianmeo/benchmark_VAE/reproducibility/'+str(args.dataset)+'/'+str(args.exp_name)+'/final_model'
    my_trained_vae = AutoModel.load_from_folder(
        model_path
    )

    evaluation_pipeline = EvaluationPipeline(
    model=my_trained_vae,
    eval_loader=eval_dataset,
    device=device,
    eval_data_with_labels=eval_dataset,
    )    

    # Generate samples and traversals
    gen_data = evaluation_pipeline.sample(
    num_samples=50,
    batch_size=10,
    output_dir=model_path,
    return_gen=True,
    labels=True,
    )
 

    # Generate t_sne csv file 
    tsne_df = evaluation_pipeline.t_sne_plot(
    labels=True,
    model_name=args.exp_name,
    )
    
    tsne_df.to_csv(f'experiments/tsne/{args.dataset}/{args.exp_name}.csv', index=False)


    #disentanglement_metrics, jemmig_metrics, results_dci = evaluation_pipeline.disentanglement_metrics(labels_flag=True, model_name=args.model_name) 
    #for key in disentanglement_metrics.keys():
    #    if key == 'WSEP':
    #        results = {'model_name': str(args.exp_name), 'WSEP': str(disentanglement_metrics['WSEP'])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEP.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEP.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEP.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEP.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #    if key=="WSEPIN":
    #        results = {'model_name': str(args.exp_name), 'WSEPIN': str(disentanglement_metrics["WSEPIN"])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEPIN.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEPIN.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEPIN.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WSEPIN.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #    if key=="WINDIN":
    #        results = {'model_name': str(args.exp_name), 'WINDIN': str(disentanglement_metrics["WINDIN"])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_WINDIN.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WINDIN.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WINDIN.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_WINDIN.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #for key in jemmig_metrics.keys():
    #    if key == 'JEMMIG':
    #        results = {'model_name': str(args.exp_name), 'JEMMIG': str(jemmig_metrics['JEMMIG'])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_JEMMIG.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_JEMMIG.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_JEMMIG.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_JEMMIG.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #    if key == 'RMIG':
    #        results = {'model_name': str(args.exp_name), 'RMIG': str(jemmig_metrics['RMIG'])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_RMIG.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_RMIG.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_RMIG.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_RMIG.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #for key in results_dci['lasso'].keys():
    #    if key == 'disentanglement':
    #        results = {'model_name': str(args.exp_name), 'disentanglement': str(results_dci['lasso']['disentanglement'])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_lasso.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_lasso.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_lasso.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_lasso.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #    if key == 'completeness':
    #        results = {'model_name': str(args.exp_name), 'completeness': str(results_dci['lasso']['completeness'])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_lasso.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_lasso.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_lasso.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_lasso.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #    if key == 'informativeness':
    #        results = {'model_name': str(args.exp_name), 'informativeness': str(results_dci['lasso']['informativeness'])}
    #        if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_lasso.json'):
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_lasso.json', "r") as jsonFile:
    #                data = json.load(jsonFile)
    #            data.append(results)
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_lasso.json', "w+") as jsonFile:
    #                json.dump(data, jsonFile)
    #        else:
    #            with open(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_lasso.json', "w+") as jsonFile:
    #                json.dump([results], jsonFile)
    #for key in results_dci['random_forest'].keys():
    #            if key == 'disentanglement':
    #                results = {'model_name': str(args.exp_name), 'disentanglement': str(results_dci['random_forest']['disentanglement'])}
    #                if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_random_forest.json'):
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_random_forest.json', "r") as jsonFile:
    #                        data = json.load(jsonFile)
    #                    data.append(results)
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_random_forest.json', "w+") as jsonFile:
    #                        json.dump(data, jsonFile)
    #                else:
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_disentanglement_random_forest.json', "w+") as jsonFile:
    #                        json.dump([results], jsonFile)
    #            if key == 'completeness':
    #                results = {'model_name': str(args.exp_name), 'completeness': str(results_dci['random_forest']['completeness'])}
    #                if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_random_forest.json'):
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_random_forest.json', "r") as jsonFile:
    #                        data = json.load(jsonFile)
    #                    data.append(results)
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_random_forest.json', "w+") as jsonFile:
    #                        json.dump(data, jsonFile)
    #                else:
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_completeness_random_forest.json', "w+") as jsonFile:
    #                        json.dump([results], jsonFile)
    #            if key == 'informativeness':
    #                results = {'model_name': str(args.exp_name), 'informativeness': str(results_dci['random_forest']['informativeness'])}
    #                if os.path.exists(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_random_forest.json'):
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_random_forest.json', "r") as jsonFile:
    #                        data = json.load(jsonFile)
    #                    data.append(results)
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_random_forest.json', "w+") as jsonFile:
    #                        json.dump(data, jsonFile)
    #                else:
    #                    with open(f'experiments/disentanglement_metrics/{args.dataset}/results_informativeness_random_forest.json', "w+") as jsonFile:
    #                        json.dump([results], jsonFile)
#
    #
    ## Compute visual scores 
    #mse, fid = evaluation_pipeline.compute_eval_scores(True)
#
    #results = {'model_name': str(args.exp_name), 'FID': str(fid), 'MSE': str(mse)}
    #if os.path.exists(f'experiments/validation_scores/{args.dataset}/visual_scores_results.json'):
    #    with open(f'experiments/validation_scores/{args.dataset}/visual_scores_results.json', "r") as jsonFile:
    #        data = json.load(jsonFile)
    #    data.append(results)
    #    with open(f'experiments/validation_scores/{args.dataset}/visual_scores_results.json', "w+") as jsonFile:
    #        json.dump(data, jsonFile)
    #else:
    #    with open(f'experiments/validation_scores/{args.dataset}/visual_scores_results.json', "w+") as jsonFile:
    #        json.dump([results], jsonFile)


if __name__ == "__main__":

    main(args)
