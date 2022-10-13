"""A collection of Neural nets used to perform the benchmark on 3Dshapes"""

from .convnets import *

__all__ = [
    
    "Encoder_Conv_VAE_3DSHAPES",
    "Equivariant_Encoder_Conv_VAE_3DSHAPES"
    "Decoder_Conv_VAE_3DSHAPES",
    "SBD_Conv_VAE_3DSHAPES",
    "SBD_masks_Conv_VAE_3DSHAPES",
    "Equivariant_SBD_Conv_VAE_3DSHAPES",
]
