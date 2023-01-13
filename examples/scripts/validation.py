import argparse
import importlib
import logging
import os
import h5py
import numpy as np
import torch
#from pythae.samplers import NormalSampler
from pythae.data.preprocessors import DataProcessor
from pythae.models import RHVAE
from pythae.models.rhvae import RHVAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import (
    BaseTrainerConfig,
    CoupledOptimizerTrainerConfig,
    AdversarialTrainerConfig,
)
from pythae.models import AutoModel
from pythae.samplers import MAFSamplerConfig
from pythae.pipelines.metrics import EvaluationPipeline
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from pythae.data.datasets import CelebADataset

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
    choices=["mnist", "cifar10", "celeba","dsprites", "3Dshapes"],
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
    "--enc_celeba",
    help='encoder celeba',
    type=bool,
    default=False,
    choices=[True, False],
)

ap.add_argument(
    "--dec_celeba",
    help='decoder celeba',
    type=bool,
    default=False,
    choices=[True, False],
)

ap.add_argument(
    "--exp_name", 
    type=str,
    help='name experiment',
    default='None',
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
    "--data_path",
    help='dataset folder path ',
    type=str,
    default="/home/cristianmeo/Datasets/",
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
    
    if args.dataset == "mnist":

        if args.nn == "convnet":

            from pythae.models.nn.benchmarks.mnist import Encoder_Conv_AE_MNIST as Encoder_AE
            from pythae.models.nn.benchmarks.mnist import Encoder_Conv_VAE_MNIST as Encoder_VAE
            from pythae.models.nn.benchmarks.mnist import Encoder_Conv_SVAE_MNIST as Encoder_SVAE
            from pythae.models.nn.benchmarks.mnist import Encoder_Conv_AE_MNIST as Encoder_VQVAE
            from pythae.models.nn.benchmarks.mnist import Decoder_Conv_AE_MNIST as Decoder_AE
            from pythae.models.nn.benchmarks.mnist import Decoder_Conv_AE_MNIST as Decoder_VQVAE

        elif args.nn == "resnet":
            from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_AE_MNIST as Encoder_AE
            from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST as Encoder_VAE
            from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_SVAE_MNIST as Encoder_SVAE
            from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VQVAE_MNIST as Encoder_VQVAE
            from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_AE_MNIST as Decoder_AE
            from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_VQVAE_MNIST as Decoder_VQVAE
        
        
        from pythae.models.nn.benchmarks.mnist import (
            Discriminator_Conv_MNIST as Discriminator,
        )

    elif args.dataset == "cifar10":
       
        from pythae.models.nn.benchmarks.cifar10 import Encoder_Conv_VAE_CIFAR10 as Encoder_VAE

        if args.dec_celeba:
            from pythae.models.nn.benchmarks.cifar10 import Decoder_Conv_VAE_CIFAR10 as Decoder_VAE
        else:
            from pythae.models.nn.benchmarks.cifar10 import SBD_Conv_VAE_CIFAR10 as Decoder_VAE

        image_size=64
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(image_size)])

        train_data = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform).data.transpose((0, 3, 1, 2))/255
    

        eval_data = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform).data.transpose((0, 3, 1, 2))/255
       
   
            
    if args.dataset == "3Dshapes": 

        from pythae.models.nn.benchmarks.shapes import Encoder_Conv_VAE_3DSHAPES as Encoder_VAE
        from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE

        dataset = h5py.File('/home/cristianmeo/Datasets/3dshapes.h5', 'r')
        
        data =  np.array(dataset['images']).transpose((0, 3, 1, 2))/ 255.0
        
        train_data = data[:int(data.shape[0]*0.9)]
        eval_data = data[int(data.shape[0]*0.9):]

    if args.dataset == "dsprites":

        from pythae.models.nn.benchmarks.dsprites import Encoder_Conv_VAE_DSPRITES as Encoder_VAE
        from pythae.models.nn.benchmarks.dsprites import SBD_Conv_VAE_DSPRITES as Decoder_VAE
        dataset = h5py.File('/home/cristianmeo/Datasets/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5', 'r')
        image_data =  np.expand_dims(np.array(dataset['imgs']), 1) / 255.0

        train_data = image_data[:int(image_data.shape[0]*0.8)]
        eval_data = image_data[int(image_data.shape[0]*0.8):]

    if args.dataset == "celeba":

        if args.enc_celeba:
            from pythae.models.nn.benchmarks.celeba import Encoder_Conv_VAE_CELEBA as Encoder_VAE
        else:
            from pythae.models.nn.benchmarks.shapes import Encoder_Conv_VAE_3DSHAPES as Encoder_VAE
        if args.dec_celeba:
            from pythae.models.nn.benchmarks.celeba import Decoder_Conv_VAE_CELEBA as Decoder_VAE
        else:
            from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE
        # C=31 Enc and Dec of Celeba 
        # C=30 later, Enc of Celeba 
        # C=32 Enc 3DShapes, Dec Celeba 

        # Spatial size of training images, images are resized to this size.
        image_size = 64
        img_folder=args.data_path+'celeba/img_align_celeba'
        # Transformations to be applied to each individual image sample
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
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

    elif args.model_name == "vae":
        from pythae.models import VAE, VAEConfig

        if args.model_config is not None:
            model_config = VAEConfig.from_json_file(args.model_config)

        else:
            model_config = VAEConfig()

        model_config.input_dim = data_input_dim

        model = VAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "iwae":
        from pythae.models import IWAE, IWAEConfig

        if args.model_config is not None:
            model_config = IWAEConfig.from_json_file(args.model_config)

        else:
            model_config = IWAEConfig()

        model_config.input_dim = data_input_dim

        model = IWAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "info_vae":
        from pythae.models import INFOVAE_MMD, INFOVAE_MMD_Config

        if args.model_config is not None:
            model_config = INFOVAE_MMD_Config.from_json_file(args.model_config)

        else:
            model_config = INFOVAE_MMD_Config()

        model_config.input_dim = data_input_dim

        model = INFOVAE_MMD(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "wae":
        from pythae.models import WAE_MMD, WAE_MMD_Config

        if args.model_config is not None:
            model_config = WAE_MMD_Config.from_json_file(args.model_config)

        else:
            model_config = WAE_MMD_Config()

        model_config.input_dim = data_input_dim

        model = WAE_MMD(
            model_config=model_config,
            encoder=Encoder_AE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "rae_l2":
        from pythae.models import RAE_L2, RAE_L2_Config

        if args.model_config is not None:
            model_config = RAE_L2_Config.from_json_file(args.model_config)

        else:
            model_config = RAE_L2_Config()

        model_config.input_dim = data_input_dim

        model = RAE_L2(
            model_config=model_config,
            encoder=Encoder_AE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "rae_gp":
        from pythae.models import RAE_GP, RAE_GP_Config

        if args.model_config is not None:
            model_config = RAE_GP_Config.from_json_file(args.model_config)

        else:
            model_config = RAE_GP_Config()

        model_config.input_dim = data_input_dim

        model = RAE_GP(
            model_config=model_config,
            encoder=Encoder_AE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "vamp":
        from pythae.models import VAMP, VAMPConfig

        if args.model_config is not None:
            model_config = VAMPConfig.from_json_file(args.model_config)

        else:
            model_config = VAMPConfig()

        model_config.input_dim = data_input_dim

        model = VAMP(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "beta_vae":
        from pythae.models import BetaVAE, BetaVAEConfig
        print(args.model_config)
        if args.model_config is not None:
            model_config = BetaVAEConfig.from_json_file(args.model_config)

        else:
            model_config = BetaVAEConfig()

        model_config.input_dim = data_input_dim

        model = BetaVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )

    elif args.model_name == "hvae":
        from pythae.models import HVAE, HVAEConfig

        if args.model_config is not None:
            model_config = HVAEConfig.from_json_file(args.model_config)

        else:
            model_config = HVAEConfig()

        model_config.input_dim = data_input_dim

        model = HVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "rhvae":
        from pythae.models import RHVAE, RHVAEConfig

        if args.model_config is not None:
            model_config = RHVAEConfig.from_json_file(args.model_config)

        else:
            model_config = RHVAEConfig()

        model_config.input_dim = data_input_dim

        model = RHVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "tc_vae":
        from pythae.models import TCVAE, TCVAEConfig

        if args.model_config is not None:
            model_config = TCVAEConfig.from_json_file(args.model_config)

        else:
            model_config = TCVAEConfig()

        model_config.input_dim = data_input_dim

        model = TCVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )


    elif args.model_name == "aae":
        from pythae.models import Adversarial_AE, Adversarial_AE_Config

        if args.model_config is not None:
            model_config = Adversarial_AE_Config.from_json_file(args.model_config)

        else:
            model_config = Adversarial_AE_Config()

        model_config.input_dim = data_input_dim

        model = Adversarial_AE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "vaegan":
        from pythae.models import VAEGAN, VAEGANConfig

        if args.model_config is not None:
            model_config = VAEGANConfig.from_json_file(args.model_config)

        else:
            model_config = VAEGANConfig()

        model_config.input_dim = data_input_dim

        model = VAEGAN(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
            discriminator=Discriminator(model_config),
        )

    elif args.model_name == "vqvae":
        from pythae.models import VQVAE, VQVAEConfig

        if args.model_config is not None:
            model_config = VQVAEConfig.from_json_file(args.model_config)

        else:
            model_config = VQVAEConfig()

        model_config.input_dim = data_input_dim

        model = VQVAE(
            model_config=model_config,
            encoder=Encoder_VQVAE(model_config),
            decoder=Decoder_VQVAE(model_config),
        )

    elif args.model_name == "msssim_vae":
        from pythae.models import MSSSIM_VAE, MSSSIM_VAEConfig

        if args.model_config is not None:
            model_config = MSSSIM_VAEConfig.from_json_file(args.model_config)

        else:
            model_config = MSSSIM_VAEConfig()

        model_config.input_dim = data_input_dim

        model = MSSSIM_VAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "svae":
        from pythae.models import SVAE, SVAEConfig

        if args.model_config is not None:
            model_config = SVAEConfig.from_json_file(args.model_config)

        else:
            model_config = SVAE()

        model_config.input_dim = data_input_dim

        model = SVAE(
            model_config=model_config,
            encoder=Encoder_SVAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "disentangled_beta_vae":
        from pythae.models import DisentangledBetaVAE, DisentangledBetaVAEConfig

        if args.model_config is not None:
            model_config = DisentangledBetaVAEConfig.from_json_file(args.model_config)

        else:
            model_config = DisentangledBetaVAEConfig()

        model_config.input_dim = data_input_dim

        model = DisentangledBetaVAE(
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

    elif args.model_name == "beta_tc_vae":
        from pythae.models import BetaTCVAE, BetaTCVAEConfig

        if args.model_config is not None:
            model_config = BetaTCVAEConfig.from_json_file(args.model_config)

        else:
            model_config = BetaTCVAEConfig()

        model_config.input_dim = data_input_dim

        model = BetaTCVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_VAE(model_config),
        )

    elif args.model_name == "vae_iaf":
        from pythae.models import VAE_IAF, VAE_IAF_Config

        if args.model_config is not None:
            model_config = VAE_IAF_Config.from_json_file(args.model_config)

        else:
            model_config = VAE_IAF_Config()

        model_config.input_dim = data_input_dim

        model = VAE_IAF(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "vae_lin_nf":
        from pythae.models import VAE_LinNF, VAE_LinNF_Config

        if args.model_config is not None:
            model_config = VAE_LinNF_Config.from_json_file(args.model_config)

        else:
            model_config = VAE_LinNF_Config()

        model_config.input_dim = data_input_dim

        model = VAE_LinNF(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
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

    if model.model_name == "RAE_L2":
        training_config = CoupledOptimizerTrainerConfig.from_json_file(
            args.training_config
        )

    elif model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE":
        training_config = AdversarialTrainerConfig.from_json_file(args.training_config)

    elif model.model_name == "VAEGAN":
        from pythae.trainers import (
            CoupledOptimizerAdversarialTrainer,
            CoupledOptimizerAdversarialTrainerConfig,
        )

        training_config = CoupledOptimizerAdversarialTrainerConfig.from_json_file(
            args.training_config
        )

    else:
        training_config = BaseTrainerConfig.from_json_file(args.training_config)

    logger.info(f"Training config: {training_config}\n")

    callbacks = []

    if args.use_wandb:
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        callbacks.append(wandb_cb)

    #pipeline = TrainingPipeline(training_config=training_config, model=model)

    #pipeline(train_data=train_data, eval_data=eval_data, callbacks=callbacks)

    #metrics = []
    ## Retrieve the trained model
    #for i in [2, 4, 6, 8, 10, 12]:
    #    my_trained_vae = AutoModel.load_from_folder(
    #    '/home/cristianmeo/benchmark_VAE/examples/scripts/reproducibility/'+str(args.dataset)+'/'+str(args.exp_name)+'/checkpoint_epoch_'+str(i)
    #    )
    #    #my_sampler_config = MAFSamplerConfig(
    #    #n_made_blocks=2,
    #    #n_hidden_in_made=3,
    #    #hidden_size=64
    #    #)
    #    ## Build the pipeline
    #    #pipe = GenerationPipeline(
    #    #model=my_trained_vae,
    #    #sampler_config=my_sampler_config
    #    #)
    #    ## Launch data generation
    #    #generted_samples = pipe(
    #    #num_samples=10,
    #    #return_gen=True, # If false returns nothing
    #    #train_data=train_data, # Needed to fit the sampler
    #    #eval_data=eval_data, # Needed to fit the sampler
    #    #training_config=BaseTrainerConfig(num_epochs=200) # TrainingConfig to use to fit the sampler
    #    #)
#
    #    # Define your sampler
    #    
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    freer_device = np.argmin(memory_available)
    device = (
        "cuda:"+str(freer_device)
        if torch.cuda.is_available() and not training_config.no_cuda
        else "cpu"
    )
#
    #    evaluation_pipeline = EvaluationPipeline(
    #    model=my_trained_vae,
    #    eval_loader=eval_data,
    #    device=device
    #    )    
#
    #    disentanglement_metrics = evaluation_pipeline.disentanglement_metrics()
    #    metrics.append(disentanglement_metrics)
#
#
    #    # Generate samples
    #    gen_data = evaluation_pipeline.sample(
    #    num_samples=50,
    #    batch_size=10,
    #    output_dir='/home/cristianmeo/benchmark_VAE/examples/scripts/reproducibility/'+str(args.dataset)+'/'+str(args.exp_name)+"/checkpoint_epoch_"+str(i),
    #    return_gen=True
    #    )
    #exp_name = 'DisentangledBetaVAE_training_2022-10-19_16-39-02'
    if args.data_path[0:5] == '/home':
        directory = '/home'
    else:
        directory = '/users'
    my_trained_vae = AutoModel.load_from_folder(
        directory+'/cristianmeo/benchmark_VAE/reproducibility/'+str(args.dataset)+'/'+str(args.exp_name)+'/final_model'
    )
        #my_sampler_config = MAFSamplerConfig(
        #n_made_blocks=2,
        #n_hidden_in_made=3,
        #hidden_size=64
        #)
        ## Build the pipeline
        #pipe = GenerationPipeline(
        #model=my_trained_vae,
        #sampler_config=my_sampler_config
        #)
        ## Launch data generation
        #generted_samples = pipe(
        #num_samples=10,
        #return_gen=True, # If false returns nothing
        #train_data=train_data, # Needed to fit the sampler
        #eval_data=eval_data, # Needed to fit the sampler
        #training_config=BaseTrainerConfig(num_epochs=200) # TrainingConfig to use to fit the sampler
        #)

        # Define your sampler


    evaluation_pipeline = EvaluationPipeline(
    model=my_trained_vae,
    eval_loader=eval_data,
    device=device
    )    
    
    #disentanglement_metrics, normalized_SEPIN = evaluation_pipeline.disentanglement_metrics()
    #metrics.append(disentanglement_metrics)
    #metrics.append(normalized_SEPIN)

    
    # Generate samples
    gen_data = evaluation_pipeline.sample(
    num_samples=50,
    batch_size=10,
    output_dir=directory+'/cristianmeo/benchmark_VAE/reproducibility/'+str(args.dataset)+'/'+str(args.exp_name)+"/final_model",
    return_gen=True
    )
    #print(metrics)

    


if __name__ == "__main__":

    main(args)
