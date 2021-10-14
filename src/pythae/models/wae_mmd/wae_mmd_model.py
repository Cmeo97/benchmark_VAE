import torch
import os

from ...models import AE
from .wae_mmd_config import WAE_MMD_Config
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOuput

from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_AE_MLP

from typing import Optional

import torch.nn.functional as F

class WAE_MMD(AE):
    """Wasserstein Autoencoder model (https://arxiv.org/pdf/1711.01558.pdf).
    
    Args:
        model_config(WAE_MMD_Config): The Autoencoder configuration seting the main parameters of the
            model

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: WAE_MMD_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None
    ):

        AE.__init__(
            self, model_config=model_config, encoder=encoder, decoder=decoder
            )

        self.model_name = "WAE_MMD"

        self.kernel_choice = model_config.kernel_choice
    
    def forward(self, inputs: BaseDataset) -> ModelOuput:
        """The input data is encoded and decoded
        
        Args:
            inputs (BaseDataset): An instance of pythae's datasets
            
        Returns:
            ModelOuput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        z = self.encoder(x).embedding
        recon_x = self.decoder(z)['reconstruction']

        z_prior = torch.randn_like(z, device=x.device)

        loss, recon_loss, mmd_loss = self.loss_function(recon_x, x, z, z_prior)

        output = ModelOuput(
            loss=loss,
            recon_loss=recon_loss,
            mmd_loss=mmd_loss,
            recon_x=recon_x,
            z=z
        )

        return output


    def loss_function(self, recon_x, x, z, z_prior):

        N = z.shape[0] # batch size

        recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='sum')

        if self.kernel_choice == 'rbf':
            k_z = self.rbf_kernel(z, z)
            k_z_prior = self.rbf_kernel(z_prior, z_prior)
            k_cross = self.rbf_kernel(z, z_prior)

        else:
            k_z = self.imq_kernel(z, z)
            k_z_prior = self.imq_kernel(z_prior, z_prior)
            k_cross = self.imq_kernel(z, z_prior)

        mmd_z = (k_z - k_z.diag()).sum() / (N - 1)
        mmd_z_prior = (k_z_prior - k_z_prior.diag()).sum() / (N - 1)
        mmd_cross = k_cross.sum() / N

        mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

        return recon_loss + self.model_config.reg_weight * mmd_loss, recon_loss, mmd_loss

    def imq_kernel(self, z1, z2):
        """Returns a matrix of shape batch X batch containing the pairwise kernel computation"""

        C = 2. * self.model_config.latent_dim * self.model_config.kernel_bandwidth ** 2

        k = C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

        return k

    def rbf_kernel(self, z1, z2):
        """Returns a matrix of shape batch X batch containing the pairwise kernel computation"""

        C = 2. * self.model_config.latent_dim * self.model_config.kernel_bandwidth ** 2

        k = torch.exp(- torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 /  C)

        return k


    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = WAE_MMD_Config.from_json_file(path_to_model_config)

        return model_config


    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:
                a ``model_config.json`` and a ``model.pt`` if no custom architectures were
                provided

                or
                a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)

        return model