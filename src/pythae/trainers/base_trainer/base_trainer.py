import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pythae.pipelines.metrics import EvaluationPipeline
from ...customexception import ModelError
from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..trainer_utils import set_seed
from ..training_callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from .base_training_config import BaseTrainerConfig
from torch.optim.lr_scheduler import LambdaLR
from pythae.debugger import Debugger
from tsne_torch import TorchTSNE as TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseTrainer:
    """Base class to perform model training.

    Args:
        model (BaseAE): A instance of :class:`~pythae.models.BaseAE` to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        eval_dataset (BaseDataset): The evaluation dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_config (BaseTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`BaseTrainerConfig` is used. Default: None.

        optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training. If None, a :class:`~torch.optim.Adam` optimizer is used. Default: None.

        scheduler (~torch.optim.lr_scheduler): An instance of `torch.optim.Optimizer` used for
            training. If None, a :class:`~torch.optim.Adam` optimizer is used. Default: None.

        callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
            A list of callbacks to use during training.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[BaseTrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        callbacks: List[TrainingCallback] = None,
        kwargs=None, 
    ):

        if training_config is None:
            training_config = BaseTrainerConfig()

        if training_config.output_dir is None:
            output_dir = "dummy_output_dir"
            training_config.output_dir = output_dir

        if not os.path.exists(training_config.output_dir):
            os.makedirs(training_config.output_dir)
            logger.info(
                f"Created {training_config.output_dir} folder since did not exist.\n"
            )
        self.update_architecture = kwargs['update_architecture']
        self.training_config = training_config
        self.name_exp = kwargs['name_exp']
        if kwargs['use_hpc'] is not None:
            self.use_hpc = kwargs['use_hpc']
        else:
            self.use_hpc = False

        set_seed(self.training_config.seed)
        if self.use_hpc:
            device = (
                "cuda"
                if torch.cuda.is_available() and not training_config.no_cuda
                else "cpu"
            )
        else:
            self.get_freer_gpu()
            device = (
                "cuda:"+str(self.freer_device)
                if torch.cuda.is_available() and not training_config.no_cuda
                else "cpu"
            )
        self.device = device
        # place model on device
        model = model.to(device)
        model.device = device

        # set optimizer
        if optimizer is None:
            optimizer = self.set_default_optimizer(model)

        else:
            optimizer = self._set_optimizer_on_device(optimizer, device)

        # set scheduler
        if scheduler is None:
            scheduler = self.set_default_scheduler(model, optimizer)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        # Define the loaders
        train_loader = self.get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            eval_loader = self.get_eval_dataloader(eval_dataset)

        else:
            logger.info(
                "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        if callbacks is None:
            callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=callbacks, model=model, optimizer=optimizer, scheduler=scheduler
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

        self.clip_gradient_norm = False
        self.gradient_clip_norm_value = 300.

        # Whether or not to use gradient skipping. This is usually unnecessary when using gradient smoothing but it doesn't hurt to keep it for safety.
        self.gradient_skip_flag = True
        # Defines the threshold at which to skip the update. Also defined for nats/dim loss.
        self.gradient_skip_threshold = 800.

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

    def get_freer_gpu(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        self.freer_device = np.argmin(memory_available)


    
  

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

    def set_default_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )

        return optimizer

    def set_default_scheduler(
        self, model: BaseAE, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler:

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, verbose=True
        )

        return scheduler

    def linear_warmup_exp_decay(self,
        warmup_steps: Optional[int] = None,
        exp_decay_rate: Optional[float] = None,
        exp_decay_steps: Optional[int] = None,
    ) -> Callable[[int], float]:
        assert (exp_decay_steps is None) == (exp_decay_rate is None)
        use_exp_decay = exp_decay_rate is not None
        if warmup_steps is not None:
            assert warmup_steps > 0

        def lr_lambda(step):
           
            multiplier = 1.0
            if warmup_steps is not None and step < warmup_steps:
                multiplier *= step / warmup_steps
            if use_exp_decay:
                multiplier *= exp_decay_rate ** (step / exp_decay_steps)
           
            return multiplier
      
        return lr_lambda


    def _run_model_sanity_check(self, model, loader):
        try:
            inputs = next(iter(loader))
            train_dataset = self._set_inputs_to_device(inputs)
            model(train_dataset)

        except Exception as e:
            raise ModelError(
                "Error when calling forward method from model. Potential issues: \n"
                " - Wrong model architecture -> check encoder, decoder and metric architecture if "
                "you provide yours \n"
                " - The data input dimension provided is wrong -> when no encoder, decoder or metric "
                "provided, a network is built automatically but requires the shape of the flatten "
                "input data.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

    def _set_optimizer_on_device(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

        return optim

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):

        inputs_on_device = inputs
        if self.use_hpc:
            if self.device == "cuda":
                cuda_inputs = dict.fromkeys(inputs)

                for key in inputs.keys():
                    if torch.is_tensor(inputs[key]):
                        cuda_inputs[key] = inputs[key].to(self.device)

                    else:
                        cuda_inputs = inputs[key]
        
            inputs_on_device = cuda_inputs
        else:
            if self.device == "cuda:"+str(self.freer_device):
                cuda_inputs = dict.fromkeys(inputs)

                for key in inputs.keys():
                    if torch.is_tensor(inputs[key]):
                        cuda_inputs[key] = inputs[key].to(self.device)

                    else:
                        cuda_inputs = inputs[key]
        
            inputs_on_device = cuda_inputs

        return inputs_on_device

    def _optimizers_step(self, model, model_output=None, epoch=None):
        
        loss = model_output.loss
        
        if model_output.reg_loss > 1000: 
            loss.backward()
            total_norm = self.gradient_clip(model)
            skip, _ = self.gradient_skip(total_norm)

            if not skip:
                self.optimizer.step()
                self.optimizer.zero_grad()
            else: 
                #self.debugger.tensorboard_log(self.model, model_output, self.optimizer, epoch, total_norm)
                print("Gradient skip")
                print("Total norm: ", total_norm)
                self.optimizer.zero_grad()

        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
        
        

       
    def _global_norm(self, model):
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            total_norm = torch.tensor(0.0)
        else:
            device = parameters[0].grad.device
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
        return total_norm

    def gradient_clip(self, model):
        if self.clip_gradient_norm:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        max_norm=self.gradient_clip_norm_value)
        else:
            total_norm = self._global_norm(model)
       
        return total_norm


    def gradient_skip(self, global_norm):
        if self.gradient_skip_flag:
            if torch.any(torch.isnan(global_norm)) or global_norm >= self.gradient_skip_threshold:
                skip = True
                gradient_skip_counter_delta = 1.

            else:
                skip = False
                gradient_skip_counter_delta = 0.

        else:
            skip = False
            gradient_skip_counter_delta = 0.

        return skip, gradient_skip_counter_delta

    def _schedulers_step(self, metrics=None):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics)

        else:
            self.scheduler.step()

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model.model_config
        )

        # run sanity check on the model
        self._run_model_sanity_check(self.model, self.train_loader)

        logger.info("Model passed sanity check !\n")

        self._training_signature = (
            str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.training_config.output_dir,
            #f"{self.model.model_name}_training_{self._training_signature}",
            self.name_exp
        )
        #training_dir = self.name_exp
        self.training_dir = training_dir
        #self.debugger = Debugger(self.training_dir, self.train_dataset, self.device)

        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            logger.info(
                f"Created {training_dir}. \n"
                "Training config, checkpoints and final model will be saved here.\n"
            )

        log_verbose = False

        # set up log file
        if log_output_dir is not None:
            log_dir = log_output_dir
            log_verbose = True

            # if dir does not exist create it
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                logger.info(f"Created {log_dir} folder since did not exists.")
                logger.info("Training logs will be recodered here.\n")
                logger.info(" -> Training can be monitored here.\n")

            # create and set logger
            log_name = f"training_logs_{self._training_signature}"

            file_logger = logging.getLogger(log_name)
            file_logger.setLevel(logging.INFO)
            f_handler = logging.FileHandler(
                os.path.join(log_dir, f"training_logs_{self._training_signature}.log")
            )
            f_handler.setLevel(logging.INFO)
            file_logger.addHandler(f_handler)

            # Do not output logs in the console
            file_logger.propagate = False

            file_logger.info("Training started !\n")
            file_logger.info(
                f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
                f" - batch_size: {self.training_config.batch_size}\n"
                f" - checkpoint saving every {self.training_config.steps_saving}\n"
            )

            file_logger.info(f"Model Architecture: {self.model}\n")
            file_logger.info(f"Optimizer: {self.optimizer}\n")

        logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        for epoch in range(1, self.training_config.num_epochs + 1):

            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = {}

            epoch_train_loss, logs, model_output = self.train_step(epoch)
            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_mse"] = logs['train_mse']
            metrics["train_kl"] = logs['train_kl']
            metrics["train_cvib"] = logs['train_cvib']

            if len(logs.keys()) > 3:
                for i in range(len(logs.keys()) - 3): 
                    metric_name = 'train_SEPIN_'+str(i)
                    metrics[metric_name] = logs[metric_name]

            if self.eval_dataset is not None:
                epoch_eval_loss, logs = self.eval_step(epoch)
                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_mse"] = logs['eval_mse']
                metrics["eval_kl"] = logs['eval_kl']
                metrics["eval_cvib"] = logs['eval_cvib']
                self._schedulers_step(epoch_eval_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(epoch_train_loss)

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            if (
                self.training_config.steps_predict is not None
                and epoch % self.training_config.steps_predict == 0
            ):
                true_data, reconstructions, generations, traversal = self.predict(best_model)



                self.callback_handler.on_prediction_step(
                    self.training_config,
                    true_data=true_data,
                    reconstructions=reconstructions,
                    generations=generations,
                    traversal=traversal,
                    global_step=epoch,
                )

            self.callback_handler.on_epoch_end(training_config=self.training_config)
            #########################################################################################################################
            #self.debugger.tensorboard_log(self.model, model_output, self.optimizer, epoch, self._global_norm(self.model), metrics)
            #########################################################################################################################
            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                self.save_checkpoint(
                    model=self._best_model, dir_path=training_dir, epoch=epoch
                )
                logger.info(f"Saved checkpoint at epoch {epoch}\n")
                #self.debugger.tensorboard_log(self.model, model_output, self.optimizer, epoch, self._global_norm(self.model), metrics)

                if log_verbose:
                    file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

                ##########################################################################
                # T-SNE plot
                #plt.figure(figsize=(16,10))
                #sns.scatterplot(
                #x="pca-one", y="pca-two",
                #hue="y",
                #palette=sns.color_palette("hls", 10),
                #data=df.loc[rndperm,:],
                #legend="full",
                #alpha=0.3
                #)

            self.callback_handler.on_log(
                self.training_config, metrics, logger=logger, global_step=epoch
            )

        final_dir = os.path.join(training_dir, "final_model")

        self.save_model(self._best_model, dir_path=final_dir)
        logger.info("Training ended!")
        logger.info(f"Saved final model in {final_dir}")

        self.callback_handler.on_train_end(self.training_config)

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
        )

        self.model.eval()

        epoch_loss = 0
        mse = 0
        kl = 0
        cvib = 0

        logs = {}
        for inputs in self.eval_loader:

            inputs = self._set_inputs_to_device(inputs)

            try:
                with torch.no_grad():

                    model_output = self.model(
                        inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
                    )

            except RuntimeError:
                model_output = self.model(
                    inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
                )

            loss = model_output.loss

            epoch_loss += loss.item()
            mse += model_output.reconstruction_loss.item()
            kl += model_output.reg_loss.item()
            cvib += model_output.cvib_loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_loss /= len(self.eval_loader)
        mse /= len(self.eval_loader)
        kl /= len(self.eval_loader)
        cvib /= len(self.eval_loader)

        logs['eval_mse'] = mse
        logs['eval_kl'] = kl
        logs['eval_cvib'] = cvib

        return epoch_loss, logs

    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        # set model in train model
        self.model.train()

        epoch_loss = 0
        mse = 0
        kl = 0
        cvib = 0

        for inputs in self.train_loader:

            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(
                inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
            )

            self._optimizers_step(self.model, model_output, epoch)
            loss = model_output.loss

            epoch_loss += loss.item()
            mse += model_output.reconstruction_loss.item()
            kl += model_output.reg_loss.item()
            cvib += model_output.cvib_loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        logs = {}
        if epoch % self.training_config.steps_saving == 0:

            evaluation_pipeline = EvaluationPipeline(
            model=self.model, device=self.device,
            eval_loader=self.eval_loader
            )   

            if self.model.latent_dim < 20:    
                print('Disentanglement Metrics at epoch: ', epoch)
                disentanglement_metrics, normalized_SEPIN = evaluation_pipeline.disentanglement_metrics()

                idx_min_SEPIN = np.argmin(normalized_SEPIN)
                min_SEPIN = normalized_SEPIN[idx_min_SEPIN]

                for i in range(normalized_SEPIN.shape[0]):
                    name_metric='train_SEPIN_'+str(i)
                    logs[name_metric] = normalized_SEPIN[i]

            #if min_SEPIN < 1e-5 and self.update_architecture:
            #    perturbations = []
            #    idxs = np.where(normalized_SEPIN<1e-5)
            #    for idx in idxs[0]:
            #       model_output_ = self.model(
            #       inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset), mask_idx=idx
            #       )
            #       perturbations.append(torch.abs(loss - model_output_.loss))
            #    pb = torch.stack(perturbations).detach().cpu().numpy()
            #    if np.min(pb) < 5:
            #        min_pb_idx = np.argmin(pb)
            #        update_idx = idxs[0][min_pb_idx]
            #        self.model.update(update_idx)
            #        self._best_model = deepcopy(self.model)
            #    else:
            #        print('architecture could not be updated, minimum perturbation applied =', np.min(pb))

        epoch_loss /= len(self.train_loader)
        mse /= len(self.train_loader)
        kl /= len(self.train_loader)
        cvib /= len(self.train_loader)

        logs['train_mse'] = mse
        logs['train_kl'] = kl
        logs['train_cvib'] = cvib

        return epoch_loss, logs, model_output

    def save_model(self, model: BaseAE, dir_path: str):
        """This method saves the final model along with the config files

        Args:
            model (BaseAE): The model to be saved
            dir_path (str): The folder where the model and config files should be saved
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save model
        model.save(dir_path)

        # save training config
        self.training_config.save_json(dir_path, "training_config")

        self.callback_handler.on_save(self.training_config)

    def save_checkpoint(self, model: BaseAE, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        self.checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # save optimizer
        torch.save(
            deepcopy(self.optimizer.state_dict()),
            os.path.join(self.checkpoint_dir, "optimizer.pt"),
        )

        # save scheduler
        torch.save(
            deepcopy(self.scheduler.state_dict()),
            os.path.join(self.checkpoint_dir, "scheduler.pt"),
        )

        # save model
        model.save(self.checkpoint_dir)

        # save training config
        self.training_config.save_json(self.checkpoint_dir, "training_config")

    def predict(self, model: BaseAE):

        model.eval()

        # with torch.no_grad():

        inputs = self.eval_loader.dataset[
            : min(self.eval_loader.dataset.data.shape[0], 10)
        ]
        inputs = self._set_inputs_to_device(inputs)

        model_out = model(inputs)
        reconstructions = model_out.recon_x.cpu().detach()
        z_enc = model_out.z
        

        #X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(z_enc)  # returns shape (n_samples, 2)

        z = torch.randn_like(z_enc)
        normal_generation = model.decoder(z).reconstruction.detach().cpu()


        mu = model_out.mu
        std = model_out.std 
        std_ = torch.randn_like(std) * std
        z_traversal = mu + std_
        delta = np.linspace(-20, 20, 11)
        traversals = []
        img_traversals = {}
        for l in range(z_traversal.shape[0]):
            for i in range(z_traversal.shape[1]):
                for j in range(11):
                    z_traversal[l, i] =  mu[l, i] + delta[j] * std_[l, i]
                    traversal = model.decoder(z_traversal[l].unsqueeze(0)).reconstruction[0].detach()
                    z_traversal[l, i] = mu[l, i] - delta[j] * std_[l, i]
                    traversals.append(traversal)
            np_imagegrid = make_grid(traversals, 11, z.shape[1]).detach()
            img_traversals[str(l)] = np_imagegrid
            traversals = []
            
      
        return inputs["data"], reconstructions, normal_generation, img_traversals  # put X_emb instead of None for T-SNE axis 
