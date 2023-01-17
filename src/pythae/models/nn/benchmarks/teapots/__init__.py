"""A collection of Neural nets used to perform the benchmark on TEAPOTS"""

from .convnets import *

__all__ = [
    
    "Encoder_Conv_VAE_TEAPOTS",
    "Equivariant_Encoder_Conv_VAE_TEAPOTS"
    "Decoder_Conv_VAE_TEAPOTS",
    "SBD_Conv_VAE_TEAPOTS",
    "SBD_masks_Conv_VAE_TEAPOTS",
    "Equivariant_SBD_Conv_VAE_TEAPOTS",
]
