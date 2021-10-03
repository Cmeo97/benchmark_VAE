"""This modules is the implementation of the Riemannian Hamiltonian VAE proposed in
(https://arxiv.org/pdf/2010.11518.pdf).

This module contains:
    - a :class:`~pythae.models.RHVAE` instance which is the implementation of the model.
    - | a :class:`~pythae.models.rhvae.rhvae_sampler.RHVAESampler` instance alowing to sample from
        the latent
      | space of such a model as proposed in (https://arxiv.org/pdf/2105.00026.pdf).

"""

from .rhvae_config import RHVAEConfig
from .rhvae_model import RHVAE

__all__ = [
  "RHVAE",
  "RHVAEConfig"]
