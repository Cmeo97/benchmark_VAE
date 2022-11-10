"""Training Callbacks for training monitoring (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

import importlib
import logging

import numpy as np
from tqdm.auto import tqdm

from .base_trainer.base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)


def wandb_is_available():
    return importlib.util.find_spec("wandb") is not None

def comet_is_available():
    return importlib.util.find_spec("comet_ml") is not None


def mlflow_is_available():
    return importlib.util.find_spec("mlflow") is not None


def rename_logs(logs):
    train_prefix = "train_"
    eval_prefix = "eval_"

    clean_logs = {}

    for metric_name in logs.keys():
        if metric_name.startswith(train_prefix):
            clean_logs[metric_name.replace(train_prefix, "train/")] = logs[metric_name]

        if metric_name.startswith(eval_prefix):
            clean_logs[metric_name.replace(eval_prefix, "eval/")] = logs[metric_name]

    return clean_logs


class TrainingCallback:
    """
    Base class for creating training callbacks"""

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of training.
        """

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of training.
        """

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a training step.
        """

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a training step.
        """

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after an evaluation phase.
        """

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a prediction phase.
        """

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        """
        Event called after logging the last logs.
        """


class CallbackHandler:
    """
    Class to handle list of Callback
    """

    def __init__(self, callbacks, model, optimizer, scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks but there one is already used."
                f" The current list of callbacks is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_init_end", training_config, **kwargs)

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_begin", training_config, **kwargs)

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_end", training_config, **kwargs)

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_begin", training_config, **kwargs)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_end", training_config, **kwargs)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_begin", training_config, **kwargs)

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_end", training_config, **kwargs)

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_begin", training_config, **kwargs)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_end", training_config)

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_evaluate", **kwargs)

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_save", training_config, **kwargs)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        self.call_event("on_log", training_config, logs=logs, **kwargs)

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_prediction_step", training_config, **kwargs)

    def call_event(self, event, training_config, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                training_config,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                **kwargs,
            )


class MetricConsolePrinterCallback(TrainingCallback):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # make it print to the console.
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)

    def on_log(self, training_config, logs, **kwargs):
        logger = kwargs.pop("logger", self.logger)

        if logger is not None:
            epoch_train_loss = logs.get("train_epoch_loss", None)
            epoch_eval_loss = logs.get("eval_epoch_loss", None)

            logger.info(
                "--------------------------------------------------------------------------"
            )
            if epoch_train_loss is not None:
                logger.info(f"Train loss: {np.round(epoch_train_loss, 4)}")
            if epoch_eval_loss is not None:
                logger.info(f"Eval loss: {np.round(epoch_eval_loss, 4)}")
            logger.info(
                "--------------------------------------------------------------------------"
            )


class ProgressBarCallback(TrainingCallback):
    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs.pop("train_loader", None)
        if train_loader is not None:
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit="batch",
                desc=f"Training of epoch {epoch}/{training_config.num_epochs}",
            )

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        eval_loader = kwargs.pop("eval_loader", None)
        if eval_loader is not None:
            self.eval_progress_bar = tqdm(
                total=len(eval_loader),
                unit="batch",
                desc=f"Eval of epoch {epoch}/{training_config.num_epochs}",
            )

    def on_train_step_end(self, training_config, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_eval_step_end(self, training_config, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, training_config, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()


class WandbCallback(TrainingCallback):  # pragma: no cover
    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

    def setup(self, training_config, name_exp, **kwargs):
        self.is_initialized = True

        model_config = kwargs.pop("model_config", None)
        #project_name = kwargs.pop("project_name", "pythae_benchmarking_vae")
        #entity_name = kwargs.pop("entity_name", None)

        training_config_dict = training_config.to_dict()

        self.run = self._wandb.init(project="Disentanglement-VAE", entity="cmeo", name=name_exp)

        if model_config is not None:
            model_config_dict = model_config.to_dict()

            self._wandb.config.update(
                {
                    "training_config": training_config_dict,
                    "model_config": model_config_dict,
                }
            )

        else:
            self._wandb.config.update({**training_config_dict})

        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    def on_train_begin(self, training_config, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, "train/global_step": global_step})

    def on_prediction_step(self, training_config, **kwargs):
        kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)

        data_to_log = []

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
        ):
            for i in range(int(len(true_data)/2)):

                data_to_log.append(
                    [
                        f"img_{i}",
                        self._wandb.Image(
                            np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    reconstructions[i].cpu().detach().numpy(), 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    generations[i].cpu().detach().numpy(), 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                    ]
                )

            val_table = self._wandb.Table(data=data_to_log, columns=column_names)

            self._wandb.log({"my_val_table": val_table})

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.run.finish()



class CometCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `comet_ml` (https://www.comet.com/site/).
    It allows users to store their configs, monitor
    their trainings and compare runs through a graphic interface. To be able use this feature
    you will need:
    - the package `comet_ml` installed in your virtual env. If not you can install it with
    .. code-block::
        $ pip install comet_ml
    """

    def __init__(self):
        if not comet_is_available():
            raise ModuleNotFoundError(
                "`comet_ml` package must be installed. Run `pip install comet_ml`"
            )

        else:
            import comet_ml

            self._comet_ml = comet_ml

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseTrainerConfig = None,
        api_key: str = None,
        project_name: str = "pythae_experiment",
        workspace: str = None,
        exp_name: str=None,
        offline_run: bool = False,
        offline_directory: str = "./",
        **kwargs,
    ):

        """
        Setup the CometCallback.
        args:
            training_config (BaseTraineronfig): The training configuration used in the run.
            model_config (BaseAEConfig): The model configuration used in the run.
            api_key (str): Your personal comet-ml `api_key`.
            project_name (str): The name of the wandb project to use.
            workspace (str): The name of your comet-ml workspace
            offline_run: (bool): Whether to run comet-ml in offline mode.
            offline_directory (str): The path to store the offline runs. They can to be
                synchronized then by running `comet upload`.
        """

        self.is_initialized = True

        training_config_dict = training_config.to_dict()

        
        experiment = self._comet_ml.Experiment(
            api_key=api_key, project_name=project_name
        )
        experiment.set_name(exp_name)
        experiment.log_other("Created from", "Cmeo97")
        

        experiment.log_parameters(
            training_config, prefix="training_config/"
        )
        experiment.log_parameters(
            model_config, prefix="model_config/"
        )

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)

        experiment = self._comet_ml.get_global_experiment()
        experiment.log_metrics(logs, step=global_step, epoch=global_step)

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        global_step = kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation", "traversal"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)
        traversal = kwargs.pop("traversal", None)
        experiment = self._comet_ml.get_global_experiment()

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
            and traversal is not None
        ):
            for i in range(int(len(true_data)/4)):

                experiment.log_image(
                    np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1),
                    name=f"{i}_truth",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(reconstructions[i].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_reconstruction",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(generations[i].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_normal_generation",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(traversal[str(i)].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_traversal",
                    step=global_step,
                )

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        experiment = self._comet_ml.config.get_global_experiment()
        experiment.end()