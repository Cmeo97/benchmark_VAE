"""Proposed convolutional neural nets architectures suited for MNIST"""

from modulefinder import Module
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder

from ....base import BaseAEConfig
from ....base.base_utils import ModelOutput
from ...base_architectures import BaseDecoder, BaseEncoder



class Encoder_Conv_VAE_DSPRITES(BaseEncoder):
    """<
    A Convolutional encoder suited for MNIST and Variational Autoencoder-based
    models. """



    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 1), nn.BatchNorm2d(128), nn.ReLU()
            )
        )


        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128, args.latent_dim)
        self.log_var = nn.Linear(128, args.latent_dim)


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
            #print(out.shape)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output



class Decoder_Conv_VAE_DSPRITES(BaseDecoder):
    """
    A Convolutional decoder suited for DSPRITES and Autoencoder-based
    models. """


    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(args.latent_dim, 128 * 4 * 4))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4,  2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )


        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(32, 1, 4, 2, padding=1),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

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

        for i in range(max_depth):
            out = self.layers[i](out)
            #print(out.shape)
            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)
                #print(out.shape)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output






class Equivariant_Encoder_Conv_VAE_DSPRITES(BaseEncoder):
    """<
    A Convolutional encoder suited for MNIST and Variational Autoencoder-based
    models. """



    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 1), nn.BatchNorm2d(128), nn.ReLU()
            )
        )


        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128, args.latent_dim)
        self.log_var = nn.Linear(128, args.latent_dim)


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
            #print(out.shape)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output



class Encoder_Conv_VAE_DSPRITES(BaseEncoder):
    """<
    A Convolutional encoder suited for MNIST and Variational Autoencoder-based
    models. """



    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 1), nn.BatchNorm2d(128), nn.ReLU()
            )
        )


        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128, args.latent_dim)
        self.log_var = nn.Linear(128, args.latent_dim)



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
            #print(out.shape)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output



class Decoder_Conv_VAE_DSPRITES(BaseDecoder):
    """
    A Convolutional decoder suited for DSPRITES and Autoencoder-based
    models. """


    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(args.latent_dim, 128 * 4 * 4))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4,  2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )


        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(32, 1, 4, 2, padding=1),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

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

        for i in range(max_depth):
            out = self.layers[i](out)
            #print(out.shape)
            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)
                #print(out.shape)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output



