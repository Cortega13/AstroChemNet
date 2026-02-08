"""Registers available preprocessing classes."""

from src.preprocessors.autoencoder import AutoencoderPreprocessor
from src.preprocessors.autoregressive import AutoregressivePreprocessor
from src.preprocessors.uclchem_grav import UclchemGravPreprocessor

PREPROCESSOR_REGISTRY = {
    "uclchem_grav": UclchemGravPreprocessor,
    "autoencoder": AutoencoderPreprocessor,
    "autoregressive": AutoregressivePreprocessor,
}
