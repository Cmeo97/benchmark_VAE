import datetime
import itertools
import logging
import os
from copy import deepcopy
from typing import List, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..base_trainer import BaseTrainer
from ..training_callbacks import TrainingCallback
from .adversarial_trainer_config import AdversarialTrainerConfig
from pythae.pipelines.metrics import EvaluationPipeline
import numpy as np

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AdversarialTrainer(BaseTrainer):
    """Trainer using distinct optimizers for the autoencoder and the discriminator.

    Args:
        model (BaseAE): The model to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_args (AdversarialTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`AdversarialTrainerConfig` is used. Default: None.

        autoencoder_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer`
            used for training the autoencoder. If None, a :class:`~torch.optim.Adam` optimizer is
            used. Default: None.

        discriminator_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer`
            used for training the discriminator. If None, a :class:`~torch.optim.Adam` optimizer is
            used. Default: None.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[AdversarialTrainerConfig] = None,
        autoencoder_optimizer: Optional[torch.optim.Optimizer] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        autoencoder_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        discriminator_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        callbacks: List[TrainingCallback] = None,
        kwargs=None, 
    ):

        BaseTrainer.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            optimizer=None,
            callbacks=callbacks,
            kwargs=kwargs, 
        )

        # set autoencoder optimizer
        if autoencoder_optimizer is None:
            autoencoder_optimizer = self.set_default_autoencoder_optimizer(model)

        else:
            autoencoder_optimizer = self._set_optimizer_on_device(
                autoencoder_optimizer, self.device
            )

        if autoencoder_scheduler is None:
            autoencoder_scheduler = self.set_default_scheduler(
                model, autoencoder_optimizer
            )

        # set discriminator optimizer
        if discriminator_optimizer is None:
            discriminator_optimizer = self.set_default_discriminator_optimizer(model)

        else:
            discriminator_optimizer = self._set_optimizer_on_device(
                discriminator_optimizer, self.device
            )

        if discriminator_scheduler is None:
            discriminator_scheduler = self.set_default_scheduler(
                model, discriminator_optimizer
            )

        self.autoencoder_optimizer = autoencoder_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.autoencoder_scheduler = autoencoder_scheduler
        self.discriminator_scheduler = discriminator_scheduler

        self.optimizer = None


    def set_default_autoencoder_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = torch.optim.Adam(
            itertools.chain(model.encoder.parameters(), model.decoder.parameters()),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.autoencoder_optim_decay,
        )

        return optimizer

    def set_default_discriminator_optimizer(
        self, model: BaseAE
    ) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.discriminator.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.discriminator_optim_decay,
        )

        return optimizer

    def _optimizers_step(self, model_output):

        autoencoder_loss = model_output.autoencoder_loss
        discriminator_loss = model_output.discriminator_loss

        self.autoencoder_optimizer.zero_grad()
        autoencoder_loss.backward(retain_graph=True)

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()

        self.autoencoder_optimizer.step()
        self.discriminator_optimizer.step()

    def _schedulers_step(self, autoencoder_metrics=None, discriminator_metrics=None):
        if isinstance(self.autoencoder_scheduler, ReduceLROnPlateau):
            self.autoencoder_scheduler.step(autoencoder_metrics)

        else:
            self.autoencoder_scheduler.step()

        if isinstance(self.discriminator_scheduler, ReduceLROnPlateau):
            self.discriminator_scheduler.step(discriminator_metrics)

        else:
            self.discriminator_scheduler.step()

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

        self.training_dir = training_dir

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

            epoch_train_loss, logs = self.train_step(epoch)
           
            metrics["train_mse"] = logs['train_mse']
            metrics["train_kl"] = logs['train_kl']
            metrics["train_cvib"] = logs['train_cvib']
            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_autoencoder_loss"] = logs['epoch_train_autoencoder_loss']
            metrics["train_discriminator_loss"] = logs['epoch_train_discriminator_loss']

            if len(logs.keys()) > 5:
                for i in range(len(logs.keys()) - 5): 
                    metric_name = 'train_SEPIN_'+str(i)
                    metrics[metric_name] = logs[metric_name]

            if self.eval_dataset is not None:
                epoch_eval_loss, logs = self.eval_step(epoch)
                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_autoencoder_loss"] = logs['epoch_eval_autoencoder_loss']
                metrics["eval_discriminator_loss"] = logs['epoch_eval_autoencoder_loss']
                metrics["eval_mse"] = logs['eval_mse']
                metrics["eval_kl"] = logs['eval_kl']
                metrics["eval_cvib"] = logs['eval_cvib']
                self._schedulers_step(
                    autoencoder_metrics=logs['epoch_eval_autoencoder_loss'],
                    discriminator_metrics=logs['epoch_eval_autoencoder_loss'],
                )

            else:
                epoch_eval_loss = best_eval_loss

                self._schedulers_step(
                    autoencoder_metrics=logs['epoch_train_autoencoder_loss'],
                    discriminator_metrics=logs['epoch_train_discriminator_loss'],
                )

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

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                self.save_checkpoint(
                    model=best_model, dir_path=training_dir, epoch=epoch
                )
                logger.info(f"Saved checkpoint at epoch {epoch}\n")

                if log_verbose:
                    file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config, metrics, logger=logger, global_step=epoch
            )

        final_dir = os.path.join(training_dir, "final_model")

        self.save_model(best_model, dir_path=final_dir)
        logger.info("----------------------------------")
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

        epoch_autoencoder_loss = 0
        epoch_discriminator_loss = 0
        epoch_loss = 0
        mse = 0
        kl = 0
        cvib = 0

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

            autoencoder_loss = model_output.autoencoder_loss
            discriminator_loss = model_output.discriminator_loss

            loss = autoencoder_loss + discriminator_loss

            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_discriminator_loss += discriminator_loss.item()
            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_autoencoder_loss /= len(self.eval_loader)
        epoch_discriminator_loss /= len(self.eval_loader)
        epoch_loss /= len(self.eval_loader)
        mse /= len(self.eval_loader)
        kl /= len(self.eval_loader)
        cvib /= len(self.eval_loader)
        logs = {}
        logs['eval_mse'] = mse
        logs['eval_kl'] = kl
        logs['eval_cvib'] = cvib
        logs['epoch_eval_autoencoder_loss'] = epoch_autoencoder_loss 
        logs['epoch_eval_discriminator_loss'] = epoch_discriminator_loss

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

        epoch_autoencoder_loss = 0
        epoch_discriminator_loss = 0
        epoch_loss = 0
        mse = 0
        cvib = 0
        kl = 0

        for inputs in self.train_loader:

            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(
                inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
            )

            self._optimizers_step(model_output)

            autoencoder_loss = model_output.autoencoder_loss
            discriminator_loss = model_output.discriminator_loss
            loss = autoencoder_loss + discriminator_loss

            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_discriminator_loss += discriminator_loss.item()
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
        epoch_autoencoder_loss /= len(self.train_loader)
        epoch_discriminator_loss /= len(self.train_loader)

        logs['train_mse'] = mse
        logs['train_kl'] = kl
        logs['train_cvib'] = cvib
        logs['epoch_train_autoencoder_loss'] =   epoch_autoencoder_loss 
        logs['epoch_train_discriminator_loss'] = epoch_discriminator_loss

        return epoch_loss, logs

    def save_checkpoint(self, model: BaseAE, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizers
        torch.save(
            deepcopy(self.autoencoder_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "autoencoder_optimizer.pt"),
        )
        torch.save(
            deepcopy(self.discriminator_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "discriminator_optimizer.pt"),
        )

        # save model
        model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
