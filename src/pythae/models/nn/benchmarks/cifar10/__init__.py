"""A collection of Neural nets used to perform the benchmark on CIFAR10"""

from .convnets import *
from .resnets import *

__all__ = [
    "Encoder_Conv_AE_CIFAR10",
    "Encoder_Conv_VAE_CIFAR10",
    "Encoder_Conv_SVAE_CIFAR10",
    "Decoder_Conv_AE_CIFAR10",
    "Discriminator_Conv_CIFAR10",
    "SBD_Conv_VAE_CIFAR10",
    "Encoder_ResNet_AE_CIFAR10",
    "Encoder_ResNet_VAE_CIFAR10",
    "Encoder_ResNet_SVAE_CIFAR10",
    "Encoder_ResNet_VQVAE_CIFAR10",
    "Decoder_ResNet_AE_CIFAR10",
    "Decoder_ResNet_VAE_CIFAR10",
    "Decoder_ResNet_VQVAE_CIFAR10",
]
