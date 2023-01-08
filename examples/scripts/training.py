import argparse
import importlib
import logging
import os
import h5py
import numpy as np
import torch
from sklearn.utils import shuffle
from pythae.data.preprocessors import DataProcessor
from pythae.data.datasets import CelebADataset
from pythae.models import RHVAE
from pythae.models.rhvae import RHVAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import (
    BaseTrainerConfig,
    CoupledOptimizerTrainerConfig,
    AdversarialTrainerConfig,
)
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

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
    choices=["mnist", "cifar10", "celeba","dsprites", "3Dshapes", "colored-dsprites"],
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
        "ae",
        "vae",
        "beta_vae",
        "iwae",
        "wae",
        "info_vae",
        "rae_gp",
        "rae_l2",
        "vamp",
        "hvae",
        "rhvae",
        "aae",
        "vaegan",
        "vqvae",
        "msssim_vae",
        "svae",
        "disentangled_beta_vae",
        "factor_vae",
        "beta_tc_vae",
        "vae_nf",
        "vae_iaf",
        "vae_lin_nf",
        "tc_vae",
    ],
    required=True,
)
ap.add_argument(
    "--model_config",
    help="path to model config file (expected json file)",
    default=None,
)
ap.add_argument(
    "--nn",
    help="neural nets to use",
    default="convnet",
    choices=["default", "convnet","resnet"]
)
ap.add_argument(
    "--training_config",
    help="path to training config_file (expected json file)",
    default=os.path.join(PATH, "configs/base_training_config.json"),
)
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--use_hpc",
    help="whether using hpc or not",
    action="store_true",
)
ap.add_argument(
    "--use_comet",
    help="whether to log the metrics in comet",
    action="store_true",
)
ap.add_argument(
    "--seed",
    help='seed',
    type=int,
    default=0,
)

ap.add_argument(
    "--latent_dim",
    help='latent dimensions ',
    type=int,
    default=10,
)

ap.add_argument(
    "--data_path",
    help='dataset folder path ',
    type=str,
    default="/home/cristianmeo/Datasets/",
)

ap.add_argument(
    "--name_exp",
    help='experiment_name',
    type=str,
    default=None,
)


ap.add_argument(
    "--C_factor",
    help='capacity factor',
    type=float,
    default=30,
)

ap.add_argument(
    "--alpha",
    help='bound balance',
    type=float,
    default=0.5,
)

ap.add_argument(
    "--beta",
    help='beta of VIB',
    type=float,
    default=1,
)

ap.add_argument(
    "--update_architecture",
    help='architecture dynamic update',
    type=bool,
    default=False,
    choices=[True, False],
)

ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="test-project",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="benchmark_team",
)

args = ap.parse_args()


def main(args):

            
    if args.dataset == "3Dshapes": 
        print(args.data_path)
        from pythae.models.nn.benchmarks.shapes import Encoder_Conv_VAE_3DSHAPES as Encoder_VAE
        from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE
        dataset = h5py.File(args.data_path+'3dshapes.h5', 'r')
        
        data =  shuffle(np.array(dataset['images']).transpose((0, 3, 1, 2))/ 255.0)
    
        train_data = data[:int(data.shape[0]*0.8)]
        eval_data = data[int(data.shape[0]*0.8):]

    if args.dataset == "dsprites":

        from pythae.models.nn.benchmarks.dsprites import Encoder_Conv_VAE_DSPRITES as Encoder_VAE
        from pythae.models.nn.benchmarks.dsprites import SBD_Conv_VAE_DSPRITES as Decoder_VAE
        #from pythae.models.nn.benchmarks.dsprites import Decoder_Conv_VAE_DSPRITES as Decoder_VAE
        dataset = h5py.File(args.data_path+'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5', 'r')
        image_data =  np.expand_dims(np.array(dataset['imgs']), 1)

        train_data = image_data[:int(image_data.shape[0]*0.8)]
        eval_data = image_data[int(image_data.shape[0]*0.8):]

    if args.dataset == "colored-dsprites":

        from pythae.models.nn.benchmarks.colored_dsprites import Encoder_Conv_VAE_CDSPRITES as Encoder_VAE
        from pythae.models.nn.benchmarks.colored_dsprites import SBD_Conv_VAE_CDSPRITES as Decoder_VAE
        #from pythae.models.nn.benchmarks.dsprites import Decoder_Conv_VAE_DSPRITES as Decoder_VAE
        train_dataset = h5py.File(args.data_path+'dsprites_train_data.h5', 'r')
        eval_dataset = h5py.File(args.data_path+'dsprites_test_data.h5', 'r')
        print(train_dataset['data'])

        train_data = shuffle(np.array(train_dataset['data']).reshape((train_dataset['data'].shape[0]*train_dataset['data'].shape[1], 32, 32, 3)).transpose((0, 3, 1, 2))/ 255.0) 
        eval_data = shuffle(np.array(eval_dataset['data']).reshape((eval_dataset['data'].shape[0]*eval_dataset['data'].shape[1], 32, 32, 3)).transpose((0, 3, 1, 2))/ 255.0) 

    if args.dataset == "celeba":

        from pythae.models.nn.benchmarks.shapes import Encoder_Conv_VAE_3DSHAPES as Encoder_VAE
        from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE
        # Spatial size of training images, images are resized to this size.
        image_size = 64
        img_folder=args.data_path+'celeba/img_align_celeba'
        # Transformations to be applied to each individual image sample
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        ])
        # Load the dataset from file and apply transformations
        data = CelebADataset(f'{img_folder}/img_align_celeba', transform)
        train_data = np.zeros((162770, 3, 64, 64), float)
        eval_data = np.zeros((182637 - 162770, 3, 64, 64), float)
        for i in range(162770):
            train_data[i] = data[i]
        for j in range(182637 - 162770):
            eval_data[j] = data[162770 + j]
        print('data loading done!')
        



    try:
        logger.info(f"\nLoading {args.dataset} data...\n")
        if args.dataset != "dsprites" and args.dataset != "3Dshapes" and args.dataset != "colored-dsprites" and args.dataset != "celeba":
            train_data = (
                np.load(os.path.join(PATH, f"data/{args.dataset}", "train_data.npz"))[
                    "data"
                ]
                / 255.0
            )
        print("train_data shape: ",train_data.shape )
        if args.dataset != "dsprites" and args.dataset != "3Dshapes" and args.dataset != "colored-dsprites" and args.dataset != "celeba":
            eval_data = (
                np.load(os.path.join(PATH, f"data/{args.dataset}", "eval_data.npz"))["data"]
                / 255.0
            )
        print("eval_data shape: ",eval_data.shape )
    except Exception as e:
        raise FileNotFoundError(
            f"Unable to load the data from 'data/{args.dataset}' folder. Please check that both a "
            "'train_data.npz' and 'eval_data.npz' are present in the folder.\n Data must be "
            " under the key 'data', in the range [0-255] and shaped with channel in first "
            "position\n"
            f"Exception raised: {type(e)} with message: " + str(e)
        ) from e

    logger.info("Successfully loaded data !\n")
    logger.info("------------------------------------------------------------")
    logger.info("Dataset \t \t Shape \t \t \t Range")
    logger.info(
        f"{args.dataset.upper()} train data: \t {train_data.shape} \t [{train_data.min()}-{train_data.max()}] "
    )
    logger.info(
        f"{args.dataset.upper()} eval data: \t {eval_data.shape} \t [{eval_data.min()}-{eval_data.max()}]"
    )
    logger.info("------------------------------------------------------------\n")

    data_input_dim = tuple(train_data.shape[1:])

    if args.model_name == "ae":
        from pythae.models import AE, AEConfig

        if args.model_config is not None:
            model_config = AEConfig.from_json_file(args.model_config)

        else:
            model_config = AEConfig()

        model_config.input_dim = data_input_dim

        model = AE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )

    elif args.model_name == "disentangled_beta_vae":
        from pythae.models import DisentangledBetaVAE, DisentangledBetaVAEConfig

        if args.model_config is not None:
            model_config = DisentangledBetaVAEConfig.from_json_file(args.model_config)

        else:
            model_config = DisentangledBetaVAEConfig()

        model_config.input_dim = data_input_dim
        model_config.C = args.C_factor
        model_config.latent_dim = args.latent_dim
        model_config.beta = args.beta

        model = DisentangledBetaVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )

    elif args.model_name == "tc_vae":
        from pythae.models import TCVAE, TCVAEConfig

        if args.model_config is not None:
            model_config = TCVAEConfig.from_json_file(args.model_config)

        else:
            model_config = TCVAEConfig()

        model_config.input_dim = data_input_dim
        model_config.alpha = args.alpha
        model_config.C = args.C_factor
        model_config.latent_dim = args.latent_dim
        model_config.beta = args.beta

        model = TCVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )

    elif args.model_name == "factor_vae":
        from pythae.models import FactorVAE, FactorVAEConfig

        if args.model_config is not None:
            model_config = FactorVAEConfig.from_json_file(args.model_config)

        else:
            model_config = FactorVAEConfig()

        model_config.input_dim = data_input_dim

        model = FactorVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )


        print(model)

    logger.info(f"Successfully build {args.model_name.upper()} model !\n")

    encoder_num_param = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    decoder_num_param = sum(
        p.numel() for p in model.decoder.parameters() if p.requires_grad
    )
    total_num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "----------------------------------------------------------------------"
    )
    logger.info("Model \t Encoder params \t Decoder params \t Total params")
    logger.info(
        f"{args.model_name.upper()} \t {encoder_num_param} \t \t {decoder_num_param}"
        f" \t \t {total_num_param}"
    )
    logger.info(
        "----------------------------------------------------------------------\n"
    )

    logger.info(f"Model config of {args.model_name.upper()}: {model_config}\n")


    if model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE":
        training_config = AdversarialTrainerConfig.from_json_file(args.training_config)

    else:
        training_config = BaseTrainerConfig.from_json_file(args.training_config)

    logger.info(f"Training config: {training_config}\n")

    callbacks = []
    if args.name_exp is None:
        name_exp = args.model_name+'-'+args.dataset+'-'+str(args.latent_dim)+'-'+str(args.C_factor)+'-'+str(args.alpha)+'-'+str(args.beta)+'-'+str(args.seed)
    else:
        name_exp = args.name_exp
    print(name_exp)
    if args.use_wandb:
        
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            name_exp=name_exp
        )

        callbacks.append(wandb_cb)

    if args.use_comet:
        # Create you callback
        from pythae.trainers.training_callbacks import CometCallback

        comet_cb = CometCallback() # Build the callback 

        # SetUp the callback 
        comet_cb.setup(
            training_config=training_config, # training config
            model_config=model_config, # model config
            api_key="qXdTq22JXVov2VvqWgZBj4eRr", # specify your comet api-key
            project_name="tc-vae", # specify your wandb project
            exp_name=name_exp,
            #offline_run=True, # run in offline mode
            #offline_directory='my_offline_runs' # set the directory to store the offline runs
        )

        callbacks.append(comet_cb) # Add it to the callbacks list

    kwargs = {}
    kwargs['beta'] = args.beta
    kwargs['C'] = args.C_factor
    kwargs['alpha'] = args.alpha
    kwargs['latent_dim'] = args.latent_dim
    kwargs['update_architecture'] = args.update_architecture
    kwargs['name_exp'] = name_exp
    if args.use_hpc:
        kwargs['use_hpc'] = True
    else:
        kwargs['use_hpc'] = None



    pipeline = TrainingPipeline(training_config=training_config, model=model, kwargs=kwargs)

    pipeline(train_data=train_data, eval_data=eval_data, callbacks=callbacks)


if __name__ == "__main__":

    main(args)
