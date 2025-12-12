"""Preprocessor registry."""

from .autoencoder import AutoencoderPreprocessor
from .autoregressive import AutoregressivePreprocessor
from .initial import InitialPreprocessor

PREPROCESSOR_REGISTRY = {
    "initial": InitialPreprocessor,
    "autoencoder": AutoencoderPreprocessor,
    "autoregressive": AutoregressivePreprocessor,
}
