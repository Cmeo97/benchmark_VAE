"""Implementation of a VAMP prior sampler as proposed in (https://arxiv.org/pdf/1705.07120.pdf).

Available models:
------------------

.. autosummary::
    ~pythae.models.VAMP
    :nosignatures:
"""

from .vamp_sampler import VAMPSampler
from .vamp_sampler_config import VAMPSamplerConfig

__all__ = ["VAMPSampler", "VAMPSamplerConfig"]
