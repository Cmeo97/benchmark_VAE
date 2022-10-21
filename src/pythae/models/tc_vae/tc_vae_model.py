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
    r"""
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

        self.model_name = "DisentangledBetaVAE"
        self.beta = model_config.beta
        self.C = model_config.C
        self.warmup_epoch = model_config.warmup_epoch
        self.alpha = model_config.alpha

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

        mu_p = (mu/(std+1e-12)).sum(dim=1)/(1/(std+1e-12)).sum(dim=1)

        std_p = (1/(std+1e-12)).sum(dim=1)

        #z = self.E_attention(mu, log_var)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld, cvib = self.loss_function(recon_x, x, mu, log_var, std, mu_p.unsqueeze(1), std_p.unsqueeze(1), epoch)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            cvib_loss=cvib,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, std, mu_p, std_p, epoch):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
  
        KLD_f = -0.5 * self.alpha *  torch.sum(1 + torch.log(torch.abs(std_p/(std+1e-12))) - (mu_p - mu).pow(2)/(std+1e-12) - std_p/(std+1e-12), dim=1)/mu.shape[1]
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        C_factor = min(epoch / (self.warmup_epoch + 1), 1)
        KLD_diff = (1 - self.alpha) * torch.abs(KLD - self.C * C_factor)


        return (
            ((mu.shape[1] - self.alpha)/mu.shape[1] * recon_loss + self.beta * KLD_diff + KLD_f).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
            KLD_f.mean(dim=0)
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def update(self, idx_to_remove):
        n = self.encoder.weights_embedding.shape[0]
        idxs_ = np.arange(n)
        idxs = np.delete(idxs_, idx_to_remove)
        print('updating architecture')
        with torch.no_grad():
            self.encoder.weights_embedding = torch.nn.Parameter(self.encoder.weights_embedding[idxs, :])
            self.encoder.weights_log_var = torch.nn.Parameter(self.encoder.weights_log_var[idxs, :])
            self.encoder.bias_embedding = torch.nn.Parameter(self.encoder.bias_embedding[idxs])
            self.encoder.bias_log_var = torch.nn.Parameter(self.encoder.bias_log_var[idxs])
            self.decoder.layers[0][0].weight = torch.nn.Parameter(self.decoder.layers[0][0].weight.data[:, idxs])
            self.decoder.layers[0][0].in_channels = n - 1
            self.decoder.pos_embedding.channels_map.weight = torch.nn.Parameter(self.decoder.pos_embedding.channels_map.weight.data[idxs])
            self.decoder.pos_embedding.channels_map.bias = torch.nn.Parameter(self.decoder.pos_embedding.channels_map.bias.data[idxs])
            self.decoder.pos_embedding.channels_map.out_channels = n - 1

  