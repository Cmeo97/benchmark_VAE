"""Proposed Neural nets architectures suited for MNIST"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder

from ....base import BaseAEConfig
from ....base.base_utils import ModelOutput
from ...base_architectures import BaseDecoder, BaseEncoder


class Encoder_Conv_AE_CIFAR10(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for CIFAR10-64 and Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Encoder_Conv_AE_CIFAR10
            >>> from pythae.models import AEConfig
            >>> model_config = AEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_AE_CIFAR10(model_config)
            >>> encoder
            ... Encoder_Conv_AE_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import AE
        >>> model = AE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoderSaved checkpoint at epoch 10
        ... TrueSaved checkpoint at epoch 10

    .. note::
Saved checkpoint at epoch 10Saved checkpSaved checkpoint at epoch 10Saved checkpoint at epoch 10oint at epoch 10
        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.
Saved checkpoint at epoch 10Saved checkpoint at epoch 10
        .. code-block::Saved checkpoint at epoch 10
Saved checkpoint at epoch 10Saved checkpSaved checkpoint at epoch 10oint at epoch 10
Saved checkpoint at epoch 10            Saved checkpoint at epoch 10 checkpoint at epoch 10... torch.Size([2, 64])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024 * 4 * 4, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


class Encoder_Conv_VAE_CIFAR10(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for CIFAR10-64 and
    Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Encoder_Conv_VAE_CIFAR10
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_VAE_CIFAR10(model_config)
            >>> encoder
            ... Encoder_Conv_VAE_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ...   (log_var): Linear(in_features=16384, out_features=64, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True


    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models
        since it outputs the embeddings and the **log** of the covariance diagonal coefficients
        of the input data under the key `embedding` and `log_covariance`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 64])
            >>> out.log_covariance.shape
            ... torch.Size([2, 64])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 32, 32)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()



        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 32, 4, 2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 64, 4, 2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
            )
        )

        

        self.layers = layers
        self.depth = len(layers)

        self.embedding_layer = nn.Linear(256 * 4, args.latent_dim)
        self.log_var_layer = nn.Linear(256 * 4, args.latent_dim)

        self.init_weights()
    
    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i][0], nn.Conv2d) or isinstance(self.layers[i][0], nn.Linear):
                nn.init.kaiming_normal_(self.layers[i][0].weight)
                nn.init.constant_(self.layers[i][0].bias.data, 0.01)
            if isinstance(self.layers[i][1], nn.Linear):
                nn.init.kaiming_normal_(self.layers[i][1].weight)
                nn.init.constant_(self.layers[i][1].bias.data, 0.01)
        nn.init.kaiming_normal_(self.embedding_layer.weight)
        nn.init.kaiming_normal_(self.log_var_layer.weight)
        nn.init.constant_(self.embedding_layer.bias.data, 0.01)
        nn.init.constant_(self.log_var_layer.bias.data, 0.01)

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
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
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
                output["embedding"] = self.embedding_layer(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var_layer(out.reshape(x.shape[0], -1))

        return output


class Encoder_Conv_VAE_CIFAR10_original(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for CIFAR10-64 and
    Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Encoder_Conv_VAE_CIFAR10
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_VAE_CIFAR10(model_config)
            >>> encoder
            ... Encoder_Conv_VAE_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ...   (log_var): Linear(in_features=16384, out_features=64, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True


    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models
        since it outputs the embeddings and the **log** of the covariance diagonal coefficients
        of the input data under the key `embedding` and `log_covariance`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 64])
            >>> out.log_covariance.shape
            ... torch.Size([2, 64])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()



        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

       

        self.layers = layers
        self.depth = len(layers)

        self.embedding_layer = nn.Linear(512 * 4 * 4, args.latent_dim)
        self.log_var_layer = nn.Linear(512 * 4 * 4, args.latent_dim)

        self.init_weights()
    
    def init_weights(self):
        for i in range(self.depth):
            if isinstance(self.layers[i][0], nn.Conv2d) or isinstance(self.layers[i][0], nn.Linear):
                nn.init.kaiming_normal_(self.layers[i][0].weight)
                nn.init.constant_(self.layers[i][0].bias.data, 0.01)
            if isinstance(self.layers[i][1], nn.Linear):
                nn.init.kaiming_normal_(self.layers[i][1].weight)
                nn.init.constant_(self.layers[i][1].bias.data, 0.01)
        nn.init.kaiming_normal_(self.embedding_layer.weight)
        nn.init.kaiming_normal_(self.log_var_layer.weight)
        nn.init.constant_(self.embedding_layer.bias.data, 0.01)
        nn.init.constant_(self.log_var_layer.bias.data, 0.01)

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
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
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
                output["embedding"] = self.embedding_layer(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var_layer(out.reshape(x.shape[0], -1))

        return output


class Encoder_Conv_SVAE_CIFAR10(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for CIFAR10-64 and Hyperspherical autoencoder
    Variational Autoencoder.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Encoder_Conv_SVAE_CIFAR10
            >>> from pythae.models import SVAEConfig
            >>> model_config = SVAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_SVAE_CIFAR10(model_config)
            >>> encoder
            ... Encoder_Conv_SVAE_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ...   (log_concentration): Linear(in_features=16384, out_features=1, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import SVAE
        >>> model = SVAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True


    .. note::

        Please note that this encoder is only suitable for Hyperspherical Variational Autoencoder
        models since it outputs the embeddings and the **log** of the concentration in the
        Von Mises Fisher distributions under the key `embedding` and `log_concentration`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 64])
            >>> out.log_concentration.shape
            ... torch.Size([2, 1])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024 * 4 * 4, args.latent_dim)
        self.log_concentration = nn.Linear(1024 * 4 * 4, 1)

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
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_concentration"] = self.log_concentration(
                    out.reshape(x.shape[0], -1)
                )

        return output


class Decoder_Conv_AE_CIFAR10(BaseDecoder):
    """
    A Convolutional decoder Neural net suited for CIFAR10-64 and Autoencoder-based
    models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Decoder_Conv_AE_CIFAR10
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> decoder = Decoder_Conv_AE_CIFAR10(model_config)
            >>> decoder
            ... Decoder_Conv_AE_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Linear(in_features=64, out_features=65536, bias=True)
            ...     )
            ...     (1): Sequential(
            ...       (0): ConvTranspose2d(1024, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (4): Sequential(
            ...       (0): ConvTranspose2d(128, 3, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 64)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 3, 64, 64])
    """

    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 32, 32)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 512 * 8 * 8)))


        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1), nn.Sigmoid()
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
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z
      
        for i in range(max_depth):
            out = self.layers[i](out)
           
            if i == 0:
                out = out.reshape(z.shape[0], 512, 8, 8)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output


class Decoder_Conv_VAE_CIFAR10(BaseDecoder):
    """
    A Convolutional decoder Neural net suited for CIFAR10-64 and Autoencoder-based
    models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Decoder_Conv_AE_CIFAR10
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> decoder = Decoder_Conv_AE_CIFAR10(model_config)
            >>> decoder
            ... Decoder_Conv_AE_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Linear(in_features=64, out_features=65536, bias=True)
            ...     )
            ...     (1): Sequential(
            ...       (0): ConvTranspose2d(1024, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (4): Sequential(
            ...       (0): ConvTranspose2d(128, 3, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 64)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 3, 64, 64])
    """

    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 256 * 4)))


        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(32, 32, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Tanh(), 
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
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)
          
            if i == 0:
                out = out.reshape(z.shape[0], 256, 2, 2)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output


class Discriminator_Conv_CIFAR10(BaseDiscriminator):
    """
    A Convolutional discriminator Neural net suited for CIFAR10.


    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.CIFAR10 import Discriminator_Conv_CIFAR10
            >>> from pythae.models import VAEGANConfig
            >>> model_config = VAEGANConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> discriminator = Discriminator_Conv_CIFAR10(model_config)
            >>> discriminator
            ... Discriminator_Conv_CIFAR10(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): Tanh()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (4): Sequential(
            ...       (0): Linear(in_features=16384, out_features=1, bias=True)
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAEGAN
        >>> model = VAEGAN(model_config=model_config, discriminator=discriminator)
        >>> model.discriminator == discriminator
        ... True
    """

    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.discriminator_input_dim = args.discriminator_input_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.Tanh()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )

        layers.append(nn.Sequential(nn.Linear(1024 * 4 * 4, 1), nn.Sigmoid()))

        self.layers = layers
        self.depth = len(layers)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the adversarial score of the input
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level.
        """

        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):

            if i == 4:
                out = out.reshape(x.shape[0], -1)

            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = out

        return output


class SBD_Conv_VAE_CIFAR10(BaseDecoder):
    """
    A Convolutional decoder suited for 3Dshapes and Autoencoder-based
    models. """


    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 32, 32)
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
        for i in range(self.depth-1):
            if isinstance(self.layers[i][0], nn.Conv2d):
                nn.init.kaiming_normal_(self.layers[i][0].weight)
                nn.init.constant_(self.layers[i][0].bias.data, 0.01)
        nn.init.xavier_normal_(self.layers[self.depth-1][0].weight)
        nn.init.constant_(self.layers[self.depth-1][0].bias.data, 0.01)    
     
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


