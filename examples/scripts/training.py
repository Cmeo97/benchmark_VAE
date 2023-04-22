import comet_ml
import argparse
import logging
import os
from pythae.pipelines import TrainingPipeline
from pythae.trainers import (
    BaseTrainerConfig,
    AdversarialTrainerConfig,
)
from torchvision import transforms, datasets
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
    choices=["mnist", "cifar10", "celeba","dsprites", "3Dshapes", "colored-dsprites", "teapots"],
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
    "--imbalance_percentage",
    help='percentage of samples to be removed for a given factor of variation',
    type=float,
    default=0,
)
ap.add_argument(
    "--update_architecture",
    help='architecture dynamic update',
    type=bool,
    default=False,
    choices=[True, False],
)
ap.add_argument(
    "--enc_celeba",
    help='encoder celeba',
    type=bool,
    default=False,
    choices=[True, False],
)
ap.add_argument(
    "--SBD",
    help='decoder celeba',
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

    print(args)       
    if args.dataset == "3Dshapes": 
        from pythae.models.nn.benchmarks.shapes import Encoder_Conv_VAE_3DSHAPES as Encoder_VAE
        from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE

        Shapes3D_PATH='/home/cristianmeo/Datasets/3dshapes.h5'
        data = Shapes3D(Shapes3D_PATH)
        data_n_split = int(data.images.shape[0]*0.8)
        train_data = data.images[:data_n_split]
        eval_data = data.images[data_n_split:]
       
    if args.dataset == "celeba":
      
        from pythae.models.nn.benchmarks.celeba import Encoder_Conv_VAE_CELEBA as Encoder_VAE
      
        if args.SBD:
            from pythae.models.nn.benchmarks.celeba import Decoder_Conv_VAE_CELEBA as Decoder_VAE
        else:
            from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE

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
        train_data = data.images[:data_n_split]
        eval_data = data.images[data_n_split:]
       
    if args.dataset == "teapots":

        from pythae.models.nn.benchmarks.teapots import Encoder_Conv_VAE_TEAPOTS as Encoder_VAE
        if args.SBD:   #It is the opposite: SBD false uses SBD, just to be consistent with previous notion
            from pythae.models.nn.benchmarks.teapots import Decoder_Conv_VAE_TEAPOTS as Decoder_VAE
        else:
            from pythae.models.nn.benchmarks.teapots import SBD_Conv_VAE_TEAPOTS as Decoder_VAE
        
        Teapots_PATH='/home/cristianmeo/Datasets/teapots/teapots.npz'
        data = Teapots(Teapots_PATH)
        data_n_split = int(data.images.shape[0]*0.8)
        train_data = data.images[:data_n_split]
        eval_data = data.images[data_n_split:]

    if args.dataset == "cifar10":
        from pythae.models.nn.benchmarks.cifar10 import Encoder_Conv_VAE_CIFAR10 as Encoder_VAE
        if args.SBD:
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


    if args.model_name == "disentangled_beta_vae":
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

    elif args.model_name == "beta_tc_vae":
        from pythae.models import BetaTCVAE, BetaTCVAEConfig

        if args.model_config is not None:
            model_config = BetaTCVAEConfig.from_json_file(args.model_config)

        else:
            model_config = BetaTCVAEConfig()

        model_config.input_dim = data_input_dim
        model_config.latent_dim = args.latent_dim
        model_config.beta = args.beta

        model = BetaTCVAE(
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

        comet_cb = CometCallback(comet_ml) # Build the callback 

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
