import os
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .tc_vae_config import TCVAEConfig

class TCVAE(VAE):
    """
    Disentangled :math:`\beta`-VAE model.

    Args:
        model_config (DisentangledBetaVAEConfig): The Variational Autoencoder configuration setting
            the main parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """
    
    def __init__(
        self,
        model_config: TCVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        assert (
            model_config.warmup_epoch >= 0
        ), f"Provide a value of warmup epoch >= 0, got {model_config.warmup_epoch}"

        self.model_name = "TCVAE"
        self.beta = model_config.beta
        self.C = model_config.C
        self.warmup_epoch = model_config.warmup_epoch
        self.alpha = model_config.alpha
        self.eps = 1e-8
        loss = TC_Bound(self.beta, self.C, self.warmup_epoch, self.alpha)
        self.loss = loss
 
    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        epoch = kwargs.pop("epoch", self.warmup_epoch)

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        mu_p = ((mu*(std+self.eps)**-1).sum(dim=1)*((1/(std+self.eps)).sum(dim=1))**-1).unsqueeze(1)

        std_p = (((std+self.eps)**-1).sum(dim=1)).unsqueeze(1)

        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld, cvib = self.loss(recon_x, x, mu, log_var, std, mu_p, std_p, epoch)
                                                           
        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            cvib_loss=cvib,
            loss=loss,
            recon_x=recon_x,
            z=z,
            mu=mu,
            std=std
        )

        return output

    def test(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        epoch = kwargs.pop("epoch", self.warmup_epoch)

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        mu_p = ((mu*(std+self.eps)**-1).sum(dim=1)*((1/(std+self.eps)).sum(dim=1))**-1).unsqueeze(1)

        std_p = (((std+self.eps)**-1).sum(dim=1)).unsqueeze(1)

        recon_x = self.decoder(mu)["reconstruction"]

        loss, recon_loss, kld, cvib = self.loss(recon_x, x, mu, log_var, std, mu_p, std_p, epoch)
                                                           
        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            cvib_loss=cvib,
            loss=loss,
            recon_x=recon_x,
            z=z,
            mu=mu,
            std=std
        )

        return output


    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        eps = torch.randn_like(std)
        return mu + eps * std, eps


class TC_Bound(torch.nn.Module):
    def __init__(
        self,
        beta,
        C,
        warmup_epoch,
        alpha
    ):
        super(TC_Bound, self).__init__()
   
        self.beta = beta
        self.C = C
        self.warmup_epoch = warmup_epoch
        self.alpha = alpha
        self.eps = 1e-8
    
    def forward(self, recon_x, x, mu, log_var, std, mu_p, std_p, epoch):


        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).sum(dim=-1)

        KLD_f = -0.5 * torch.sum(1 + torch.log(torch.abs(std_p)+self.eps) - torch.log(torch.abs((std)+self.eps)) - (mu_p - mu).pow(2)*(std+self.eps)**-1 - std_p*(std+self.eps)**-1, dim=1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        C_factor = min(epoch / (self.warmup_epoch + 1), 1)
        KLD_diff = torch.abs(KLD - self.C * C_factor)
        KLD_f_diff = torch.abs(KLD_f - self.C * C_factor)
        K = mu.shape[1]
        return (
            #((mu.shape[1] - self.alpha)/mu.shape[1] * recon_loss + (1 - self.alpha) * self.beta * KLD_diff + self.alpha/mu.shape[1] * KLD_f_diff).mean(dim=0),
            (recon_loss + (1 - self.alpha)/(1 - self.alpha/K) * self.beta * KLD_diff + self.alpha*K/(K - self.alpha) * KLD_f_diff).mean(dim=0),
            recon_loss.mean(dim=0),
            (1 - self.alpha)/(1 - self.alpha/K) * self.beta * KLD.mean(dim=0),
            self.alpha*K/(K - self.alpha)  * KLD_f.mean(dim=0),
        )

    
     

    