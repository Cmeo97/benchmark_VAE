"""Proposed convolutional neural nets architectures suited for MNIST"""

from typing import List

import torch
from torch import Tensor
import torch.nn as nn
from pythae.models.nn import BaseDecoder, BaseEncoder

from ....base import BaseAEConfig
from ....base.base_utils import ModelOutput
from ...base_architectures import BaseDecoder, BaseEncoder

from math import sqrt
from typing import List, Optional
import torch.nn.functional as F




class Decoder_Conv_VAE_3DSHAPES(BaseDecoder):
    """
    A Convolutional decoder suited for DSPRITES and Autoencoder-based
    models. """


    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Linear(args.latent_dim, 64 * 2 * 2))


        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
                nn.ReLU(),
            )
        )


        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1)
            )
        )

        self.layers = layers
        self.depth = len(layers)
        self.init_weights()
    
    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i][0], nn.Conv2d) or isinstance(self.layers[i][0], nn.Linear):
                self.layers[i][0].weight.data.normal_(0, 0.01)
                nn.init.constant_(self.layers[i][0].bias.data, 0.0001)
       

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z
        #print('decoder shapes')
        for i in range(max_depth):
            out = self.layers[i](out)
            #print(out.shape)

            if i == 0:
                out = out.reshape(z.shape[0], 64, 2, 2)
                #print(out.shape)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output





class Encoder_Conv_VAE_3DSHAPES(BaseEncoder):
    """<
    A Convolutional encoder suited for 3DShapes and Variational Autoencoder-based
    models. """



    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.n_channels,  out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=1024, out_features=256),
                nn.ReLU(),
            )
        )


        self.layers = layers
        self.depth = len(layers)

        self.weights_embedding = torch.nn.Parameter(torch.randn(self.latent_dim, 256))
        self.weights_log_var = torch.nn.Parameter(torch.randn(self.latent_dim, 256))
        self.bias_embedding = torch.nn.Parameter(torch.randn(self.latent_dim))
        self.bias_log_var = torch.nn.Parameter(torch.randn(self.latent_dim))
   
        self.init_weights()
    
    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i][0], nn.Conv2d) or isinstance(self.layers[i][0], nn.Linear):
                self.layers[i][0].weight.data.normal_(0, 0.01)
                nn.init.constant_(self.layers[i][0].bias.data, 0.001)
        self.weights_log_var.data.normal_(0, 0.01)
        self.weights_embedding.data.normal_(0, 0.01)
        nn.init.constant_(self.bias_embedding.data, 0.001)
        nn.init.constant_(self.bias_log_var.data, 0.001)
       
        


    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i`
            where i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = F.linear(out, self.weights_embedding, self.bias_embedding)
                output["log_covariance"] = F.linear(out, self.weights_log_var, self.bias_log_var)
              
            

        return output


class SBD_Conv_VAE_3DSHAPES(BaseDecoder):
    """
    A Convolutional decoder suited for 3Dshapes and Autoencoder-based
    models. """


    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = self.input_dim[0]
        self.w_broadcast = self.input_dim[1]
        self.h_broadcast = self.input_dim[2]
      

        self.pos_embedding = PositionalEmbedding(self.input_dim[1], self.input_dim[2], self.latent_dim)
     
        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.latent_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=self.n_channels,  kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.init_weights()
    
    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i][0], nn.Conv2d) or isinstance(self.layers[i][0], nn.Linear):
                self.layers[i][0].weight.data.normal_(0, 0.01)
                nn.init.constant_(self.layers[i][0].bias.data, 0.001)
            
     
    def spatial_broadcast(self, z: Tensor) -> Tensor:
        z = z.unsqueeze(-1).unsqueeze(-1)
        return z.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()
        z = self.spatial_broadcast(z)
        z = self.pos_embedding(z)
        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z
        
        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output



class SBD_masks_Conv_VAE_3DSHAPES(BaseDecoder):
    """
    A Convolutional decoder suited for DSPRITES and Autoencoder-based
    models. """


    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = self.input_dim[0]
        self.w_broadcast = self.input_dim[1]
        self.h_broadcast = self.input_dim[2]
      

        self.pos_embedding = PositionalEmbedding(self.input_dim[1], self.input_dim[2], 1)
     
        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=self.n_channels+1,  kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.init_weights()
    
    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i][0], nn.Conv2d) or isinstance(self.layers[i][0], nn.Linear):
                self.layers[i][0].weight.data.normal_(0, 0.01)
                nn.init.constant_(self.layers[i][0].bias.data, 0)
            
     
    def spatial_broadcast(self, z: Tensor) -> Tensor:
        z = z.unsqueeze(-1).unsqueeze(-1)
        return z.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()
        bs = z.shape[0]
        z = z.unsqueeze(2).flatten(0, 1)
        z = self.spatial_broadcast(z)
        z = self.pos_embedding(z)
        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z
        
        for i in range(max_depth):
            out = self.layers[i](out)
            #print(out.shape)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                
                images, masks = out[:, :3], out[:, -1:]
                images = images.view(bs, self.latent_dim, 3, self.input_dim[1], self.input_dim[2])
                masks = masks.view(bs, self.latent_dim, 1, self.input_dim[1], self.input_dim[2])
                masks = masks.softmax(dim=1)
                image_masked = images * masks
                output["reconstruction"] = image_masked.sum(dim=1)

        return output




class InputAttentionModule(nn.Module):
    def __init__(self, latent_dim, input_dim, dim, iters=3, eps=1e-8):
       
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps
        self.scale = dim**-0.5
        self.iters = iters
        self.latent_mu = nn.Parameter(torch.rand(1, 10, dim))
        self.latent_log_sigma = nn.Parameter(torch.randn(1, 10, dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + dim))
            torch.nn.init.uniform_(self.latent_mu, -limit, limit)
            torch.nn.init.uniform_(self.latent_log_sigma, -limit, limit)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(input_dim, dim, bias=False)
        self.to_v = nn.Linear(input_dim, dim, bias=False)

        hidden_dim = max(dim, latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.gru = nn.GRUCell(dim, dim)

        self.norm_input = nn.LayerNorm(input_dim, eps=0.001)
        self.norm_latent = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)
        self.dim = dim

    def forward(self, inputs: Tensor, latent_dim: Optional[int] = None) -> Tensor:
        b, n, _ = inputs.shape
        if latent_dim is None:
            latent_dim = self.latent_dim

        mu = self.latent_mu.expand(b, latent_dim, 2)
        sigma = self.latent_log_sigma.expand(b, latent_dim, 2).exp()
        latent = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self.iters):
            latent_prev = latent
            latent = self.norm_latent(latent)
            q = self.to_q(latent)
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            #attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn)
            latent = self.gru(
                updates.reshape(-1, self.dim), latent_prev.reshape(-1, self.dim)
            )
            latent = latent.reshape(b, -1, self.dim)
            latent = latent + self.mlp(self.norm_pre_ff(latent))

        return latent


class PositionalEmbedding(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x





