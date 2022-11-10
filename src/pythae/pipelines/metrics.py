import logging
from typing import List, Optional, Union
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from ..customexception import DatasetError
from ..data.preprocessors import BaseDataset, DataProcessor
from ..models import BaseAE
from ..trainers import *
from ..trainers.training_callbacks import TrainingCallback
from .base_pipeline import Pipeline
from pythae.DisentanglementMetrics import estimate_JEMMIG_cupy, estimate_SEP_cupy, estimate_SEPIN_torch
from pythae.trainers.trainer_utils import set_seed
import torchvision
import matplotlib.pyplot as plt
from imageio import imwrite
logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class EvaluationPipeline(Pipeline):
    """
    This Pipeline provides an end to end way to train your VAE model.
    The trained model will be saved in ``output_dir`` stated in the
    :class:`~pythae.trainers.BaseTrainerConfig`. A folder
    ``training_YYYY-MM-DD_hh-mm-ss`` is
    created where checkpoints and final model will be saved. Checkpoints are saved in
    ``checkpoint_epoch_{epoch}`` folder (optimizer and training config
    saved as well to resume training if needed)
    and the final model is saved in a ``final_model`` folder. If ``output_dir`` is
    None, data is saved in ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss`` is created.

    Parameters:

        model (Optional[BaseAE]): An instance of :class:`~pythae.models.BaseAE` you want to train.
            If None, a default :class:`~pythae.models.VAE` model is used. Default: None.

        training_config (Optional[BaseTrainerConfig]): An instance of
            :class:`~pythae.trainers.BaseTrainerConfig` stating the training
            parameters. If None, a default configuration is used.
    """

    def __init__(
        self,
        model: Optional[BaseAE],
        eval_loader: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset],
        device
    ):

        set_seed(0)
        #n_device = self.get_freer_gpu()
        self.device = device
        #self.device = (
        #    "cuda:"+str(n_device)
        #    if torch.cuda.is_available() 
        #    else "cpu"
        #)
        print("selected_device : ", self.device)
        self.data_processor = DataProcessor()
        self.model = model.to(self.device)
        print(self.model)

        if isinstance(eval_loader, torch.utils.data.Dataset):
            print('evaluation_loader_loaded')
        else:

        #if isinstance(train_data, np.ndarray) or isinstance(train_data, torch.Tensor):
#
        #    logger.info("Preprocessing train data...")
        #    train_data = self.data_processor.process_data(train_data)
        #    train_dataset = self.data_processor.to_dataset(train_data)
#
        #else:
        #    train_dataset = train_data

        #if eval_data is not None:
            if isinstance(eval_loader, np.ndarray) or isinstance(eval_loader, torch.Tensor):
                logger.info("Preprocessing eval data...\n")
        
                eval_data = self.data_processor.process_data(eval_loader)
                eval_dataset = self.data_processor.to_dataset(eval_data)
                self.eval_loader = self.get_eval_dataloader(eval_dataset)
            else:
                self.eval_loader = eval_loader
        

        
       

    def disentanglement_metrics(
        self
    ):
        """
        Launch the model training on the provided data.

        Args:
            training_data (Union[~numpy.ndarray, ~torch.Tensor]): The training data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...)

            eval_data (Optional[Union[~numpy.ndarray, ~torch.Tensor]]): The evaluation data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...). If None, only uses train_fata for training. Default: None.

            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """
        
        #    else:
        #        eval_dataset = eval_data
        #else:
        #    eval_dataset = None

        # Define the loaders
        #train_loader = self.get_train_dataloader(train_dataset)

        #if eval_dataset is not None:
           #eval_loader = self.get_eval_dataloader(eval_dataset)

        all_z_mean = []
        all_z_std = []
        for inputs in self.eval_loader:

            inputs = self._set_inputs_to_device(inputs)

            x = inputs["data"]
            #print(x.shape)
            encoder_output = self.model.encoder(x)

            mu, log_var = encoder_output.embedding.detach().cpu().numpy(), encoder_output.log_covariance

            std = torch.exp(0.5 * log_var).detach().cpu().numpy()
            all_z_mean.append(mu)
            all_z_std.append(std)
        

        z_mean = np.concatenate(all_z_mean, 0)
        z_std = np.concatenate(all_z_std, 0)
        #Disentanglement Metrics
    
        #results_JEMMIG = estimate_JEMMIG_cupy(z_mean, z_std, num_samples=10000,
        #                      batch_size=5, gpu=0)
        #results_SEP = estimate_SEP_cupy(z_mean, z_std, num_samples=10000,
        #                      batch_size=5, gpu=0)
        results_SEPIN = estimate_SEPIN_torch(z_mean, z_std, self.device, num_samples=10000,
                          batch_size=250)
        #Sep_zi = np.array(results_SEPIN['SEP_zi'])
        #wandb.log({
        #'Sep_zi': Sep_zi,
        #}, step=epoch)

        return results_SEPIN


    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
    ) -> torch.Tensor:
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated
                data. Default: True.
            save_sampler_config (bool): Whether to save the sampler config. It is saved in
                output_dir.

        Returns:
            ~torch.Tensor: The generated images
        """
        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size



        x_gen_list = []
        latent_dim = self.model.encoder.latent_dim
        for i in range(full_batch_nbr):
            z = torch.randn(batch_size, latent_dim).to(self.device)
            x_gen = self.model.decoder(z)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir+'/generated_images', "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = torch.randn(batch_size, latent_dim).to(
                self.device
            )
            
            x_gen = self.model.decoder(z)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(last_batch_samples_nbr):
                    self.save_img(
                        x_gen[j],
                        output_dir+'/generated_images',
                        "%08d.png" % int(batch_size * full_batch_nbr + j),
                    )
        c=0
        for inputs in self.eval_loader:

            inputs = self._set_inputs_to_device(inputs)

            x = inputs["data"]
            #print(x.shape)
            encoder_output = self.model.encoder(x)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            
            c += 1
            if c==30:

                break


        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)

        images = []
        delta = np.linspace(-25, 25, 11)
        for l in range(10):
            for i in range(z.shape[1]):
                for j in range(11):
                    z[l*25, i] = z[l*25, i] + delta[j]*std[l*25, i]
                    x_gen = self.model.decoder(z[l*25].unsqueeze(0))["reconstruction"].detach()
                    images.append(x_gen[0])
                    z[l*25, i] = z[l*25, i] - delta[j]*std[l*25, i]
            print('traversals')
            np_imagegrid = torchvision.utils.make_grid(images, 11, z.shape[1]).detach()
            self.save_img(np_imagegrid.permute(1, 2, 0), output_dir+'/traversals/', 'traversals_'+str(l)+'.png', True)
            images = []

        x_gen_list.append(x_gen)

        if save_sampler_config:
            self.save(output_dir)

        if return_gen:
            return torch.cat(x_gen_list, dim=0)






    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:

        return DataLoader(
            dataset=train_dataset,
            batch_size=1000,
            shuffle=True,
        )

    def get_freer_gpu(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        self.freer_device = np.argmin(memory_available)
        return self.freer_device

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=eval_dataset,
            batch_size=256,
            shuffle=False,
        )

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):

        inputs_on_device = inputs

        if 'cuda' in self.device:
            cuda_inputs = dict.fromkeys(inputs)

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    cuda_inputs[key] = inputs[key].to(self.device)

                else:
                    cuda_inputs = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device

    def save_img(self, img_tensor: torch.Tensor, dir_path: str, img_name: str, flag: bool = False):
        """Saves a data point as .png file in dir_path with img_name as name.

        Args:
            img_tensor (torch.Tensor): The image of shape CxHxW in the range [0-1]
            dir_path (str): The folder where in which the images must be saved
            ig_name (str): The name to apply to the file containing the image.
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"--> Created folder {dir_path}. Images will be saved here")
        if flag:
            img = 255.0 * img_tensor.cpu().detach().numpy()
        else:
            img = 255.0 * torch.movedim(img_tensor, 0, 2).cpu().detach().numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, repeats=3, axis=-1)

        img = img.astype("uint8")
        imwrite(os.path.join(dir_path, f"{img_name}"), img)