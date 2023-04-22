import logging
from typing import List, Optional, Union
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import pandas as pd
from sklearn.manifold import TSNE
from plotnine import *
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from ..customexception import DatasetError
from ..data.preprocessors import BaseDataset, DataProcessor
from ..models import BaseAE
from ..trainers import *
from ..trainers.training_callbacks import TrainingCallback
from .base_pipeline import Pipeline
from pythae.DisentanglementMetrics import estimate_JEMMIG_torch, estimate_SEP_cupy, estimate_SEPIN_torch_full
from pythae.trainers.trainer_utils import set_seed
import torchvision
import matplotlib.pyplot as plt
from pythae.fid_score import calculate_fid_given_tensors
from imageio import imwrite
from pythae.dci import dci
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
        device, 
        eval_data_with_labels = None
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

        if eval_data_with_labels is not None: 
            self.eval_loader = self.get_eval_dataloader(eval_data_with_labels)
        else:
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

                if isinstance(eval_loader, np.ndarray) or isinstance(eval_loader, torch.Tensor):
                    logger.info("Preprocessing eval data...\n")

                    eval_data = self.data_processor.process_data(eval_loader)
                    eval_dataset = self.data_processor.to_dataset(eval_data)
                    self.eval_loader = self.get_eval_dataloader(eval_dataset)
                    print('evaluation_loader_loaded')
                else:
                    self.eval_loader = eval_loader
                    print('evaluation_loader_loaded')
            
        

        
       

    def disentanglement_metrics(self, labels_flag=False, model_name=None):
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
        all_y = []
        all_mu = []
        if labels_flag:
            for inputs, labels in self.eval_loader:

                inputs = inputs.to(self.device)

                x = inputs
                #print(x.shape)
                encoder_output = self.model.encoder(x)

                mu, log_var = encoder_output.embedding.detach().cpu().numpy(), encoder_output.log_covariance

                std = torch.exp(0.5 * log_var).detach().cpu().numpy()

                all_z_mean.append(mu)
                all_z_std.append(std)
                all_y.append(labels)
                all_mu.append(mu)

            y = np.concatenate(all_y, 0)
        else: 
            for inputs in self.eval_loader:

                inputs = self._set_inputs_to_device(inputs)

                x = inputs["data"]
                #print(x.shape)
                encoder_output = self.model.encoder(x)

                mu, log_var = encoder_output.embedding.detach().cpu().numpy(), encoder_output.log_covariance

                std = torch.exp(0.5 * log_var).detach().cpu().numpy()

                all_z_mean.append(mu)
                all_z_std.append(std)
                all_mu.append(mu)

              

        z_mean = np.concatenate(all_z_mean, 0)
        z_std = np.concatenate(all_z_std, 0)
       
        
        results = estimate_SEPIN_torch_full(z_mean, z_std, self.device, num_samples=10000,
                          batch_size=250)
        if labels_flag:
            results_jemmig = estimate_JEMMIG_torch(z_mean, z_std, y, num_samples=10000, batch_size=250, device=self.device)   
            
        all_mu = np.concatenate(all_mu, 0)
        results_dci = {}
        if labels_flag:
          
            print('calculating DCI')
            for reg_model in ['lasso', 'random_forest']:
                # First parameter in latent space is dummy and therefore ignored.
                if model_name=='TorusVAE' :
                    disentanglement, completeness, informativeness, _ = dci(y, all_mu[:, 1:], model=reg_model)
                else:
                    disentanglement, completeness, informativeness, _ = dci(y, all_mu, model=reg_model)
                
                results_dci[reg_model] = {'disentanglement': disentanglement.item(), 'completeness': completeness.item(),
                           'informativeness': informativeness.item()}
        
                
                

        if labels_flag:
            return results, results_jemmig, results_dci
        else:
            return results


    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
        labels: bool = False,
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
        
        c=0
        if labels:
            for inputs, _ in self.eval_loader:

                inputs = inputs.to(self.device)
                encoder_output = self.model.encoder(inputs)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                c += 1
                if c==30:

                    break
        else: 
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


    def t_sne_plot(
        self,
        model_name: str = None,
        labels: bool = False,
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
        

        c = 0
        if labels:
            for inputs, _ in self.eval_loader:

                inputs = inputs.to(self.device)
                encoder_output = self.model.encoder(inputs)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                c += 1
                if c==5:

                    break
        else: 
            for inputs in self.eval_loader:

                inputs = self._set_inputs_to_device(inputs)

                x = inputs["data"]
                #print(x.shape)
                encoder_output = self.model.encoder(x)

                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                c += 1
                if c==5:

                    break



        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)

        z_tsne = []
        delta = np.linspace(-25, 25, 11)
        for l in range(100):
            for i in range(z.shape[1]):
                for j in range(11):
                    z[l, i] = z[l, i] + delta[j]*std[l, i]
                    z_tsne.append(z[l].detach().cpu().numpy())
                    z[l, i] = z[l, i] - delta[j]*std[l, i]
         
        latent_space = np.concatenate(z_tsne, 0)
        tsne_latent_space = TSNE(n_components=2).fit_transform(latent_space.reshape(-1, 10))
        tsne_df = pd.DataFrame(tsne_latent_space, columns=['x1', 'x2'])
        
        #loaded_df = pd.read_csv("data.csv")
        plot = (ggplot(tsne_df)
        + geom_point(aes(x='x1', y='x2'), size = 0.5)
        + ggtitle(f't-SNE Plot of Latent Space: {model_name}')
        + theme(axis_text=element_text(size=8))
        + theme(axis_title=element_text(size=10, weight='bold'))
        + theme(plot_title=element_text(size=14, weight='bold'))
        )

        # Save the plot
        ggsave(plot, f'tsne_plot_{model_name}.png', dpi = 300)

        
        return tsne_df


    def compute_eval_scores(
        self,
        labels: bool = False,
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
        imgs = []
        true_imgs = []
        mse = 0
        if labels:
            i = 0
            for inputs, _ in self.eval_loader:

                inputs = inputs.to(self.device)
                x = {"data": inputs}
                model_output = self.model(x)

                mse += model_output.reconstruction_loss.item()

                true_imgs.append(inputs)
                imgs.append(model_output.recon_x)
                print(i)
                i += 1
                if i == 5:
                    break

                
        else: 
            for inputs in self.eval_loader:

                inputs = self._set_inputs_to_device(inputs)

                x = inputs["data"]
          
                model_output = self.model(x)

                mse += model_output.reconstruction_loss.item()
                true_imgs.append(x)
                imgs.append(model_output.recon_x)

        mse = mse / 4000
        
        FID_score_2048 = calculate_fid_given_tensors(torch.cat(true_imgs, 0), torch.cat(imgs, 0), device='cuda', batch_size=32, dims=2048).item()  

        return mse, FID_score_2048
              




        



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