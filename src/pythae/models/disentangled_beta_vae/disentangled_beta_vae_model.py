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
from .disentangled_beta_vae_config import DisentangledBetaVAEConfig


class DisentangledBetaVAE(VAE):
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
        model_config: DisentangledBetaVAEConfig,
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
        #if 'mask_idx' in kwargs.keys():
        #    self.store_parameters()
        #    self.apply_parameters_mask(kwargs['mask_idx'])    

        #z = self.E_attention(mu, log_var)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z, epoch)
        #if 'mask_idx' in kwargs.keys():
        #    self.restore_parameters()
        
        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            cvib_loss=kld*0.00,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, epoch):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)


        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        C_factor = min(epoch / (self.warmup_epoch + 1), 1)
        KLD_diff = torch.abs(KLD - self.C * C_factor)

        return (
            (recon_loss + self.beta * KLD_diff).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
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

    def apply_parameters_mask(self, mask_idx):
        
        self.encoder.weights_embedding.data[mask_idx]                 = self.encoder.weights_embedding.data[mask_idx]                 * 0
        self.encoder.weights_log_var.data[mask_idx]                   = self.encoder.weights_log_var.data[mask_idx]                   * 0
        self.encoder.bias_embedding.data[mask_idx]                    = self.encoder.bias_embedding.data[mask_idx]                    * 0
        self.encoder.bias_log_var.data[mask_idx]                      = self.encoder.bias_log_var.data[mask_idx]                      * 0
        self.decoder.layers[0][0].weight.data[:, mask_idx]            = self.decoder.layers[0][0].weight.data[:, mask_idx]            * 0
        self.decoder.pos_embedding.channels_map.weight.data[mask_idx] = self.decoder.pos_embedding.channels_map.weight.data[mask_idx] * 0
        self.decoder.pos_embedding.channels_map.bias.data[mask_idx]   = self.decoder.pos_embedding.channels_map.bias.data[mask_idx]   * 0

        with torch.no_grad():
            self.encoder.weights_embedding = torch.nn.Parameter(self.encoder.weights_embedding.data)
            self.encoder.weights_log_var = torch.nn.Parameter(self.encoder.weights_log_var.data)
            self.encoder.bias_embedding = torch.nn.Parameter(self.encoder.bias_embedding.data)
            self.encoder.bias_log_var = torch.nn.Parameter(self.encoder.bias_log_var.data)
            self.decoder.layers[0][0].weight = torch.nn.Parameter(self.decoder.layers[0][0].weight.data)
            self.decoder.pos_embedding.channels_map.weight = torch.nn.Parameter(self.decoder.pos_embedding.channels_map.weight.data)
            self.decoder.pos_embedding.channels_map.bias = torch.nn.Parameter(self.decoder.pos_embedding.channels_map.bias.data)


    def store_parameters(self):
       self.stored_weights_embedding  =  self.encoder.weights_embedding 
       self.stored_weights_log_var =  self.encoder.weights_log_var
       self.stored_bias_embedding  =  self.encoder.bias_embedding 
       self.stored_bias_log_var  =  self.encoder.bias_log_var 
       self.stored_layers_weight  =  self.decoder.layers[0][0].weight 
       self.stored_layers_in_channels  =  self.decoder.layers[0][0].in_channels 
       self.stored_pos_embedding_channels_map_weight =  self.decoder.pos_embedding.channels_map.weight
       self.stored_pos_embedding_channels_map_bias =  self.decoder.pos_embedding.channels_map.bias
       self.stored_pos_embedding_channels_map_out_channels  =  self.decoder.pos_embedding.channels_map.out_channels 

    def restore_parameters(self):
       self.encoder.weights_embedding = self.stored_weights_embedding                            
       self.encoder.weights_log_var = self.stored_weights_log_var                              
       self.encoder.bias_embedding = self.stored_bias_embedding                               
       self.encoder.bias_log_var = self.stored_bias_log_var                                 
       self.decoder.layers[0][0].weight = self.stored_layers_weight                                
       self.decoder.layers[0][0].in_channels = self.stored_layers_in_channels                           
       self.decoder.pos_embedding.channels_map.weight = self.stored_pos_embedding_channels_map_weight            
       self.decoder.pos_embedding.channels_map.bias = self.stored_pos_embedding_channels_map_bias              
       self.decoder.pos_embedding.channels_map.out_channels = self.stored_pos_embedding_channels_map_out_channels      


    def E_attention(self, latent_mu: Tensor, latent_log_var: Tensor, eps=1e-8, latent_dim=10) -> Tensor:
        b, n = latent_mu.shape
        
        mu = latent_mu.unsqueeze(1).expand(b, latent_dim, n) # [b, latent_dim, current dim ]
        std =torch.exp(0.5* latent_log_var.unsqueeze(1).expand(b,  latent_dim, n))  #[b, latent_dim, current dim]
        #latent = torch.normal(mu, sigma)   # reparametrization -> [b, latent_dim]
      
        z = mu + std * torch.randn_like(std) # #[b, latent_dim, current dim]
  
        dots = torch.einsum("bid,bjd->bij", latent_mu.unsqueeze(2), z) * latent_mu.shape[1]**-0.5
        attn = dots.softmax(dim=1) + eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        latent = torch.einsum("bjd,bij->bid", latent_mu.unsqueeze(2), attn).squeeze(2)
        #latent = self.gru(
        #    updates.reshape(-1, self.dim), latent_prev.reshape(-1, self.dim)
        #)
        #latent = latent.reshape(b, -1, self.dim)
        #latent = latent + self.mlp(self.norm_pre_ff(latent))

        return latent
