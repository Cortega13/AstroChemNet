"""Registers available preprocessing classes."""

from src.preprocessors.autoencoder import AutoencoderPreprocessor
from src.preprocessors.autoregressive import AutoregressivePreprocessor
from src.preprocessors.initial import InitialPreprocessor

PREPROCESSOR_REGISTRY = {
    "initial": InitialPreprocessor,
    "autoencoder": AutoencoderPreprocessor,
    "autoregressive": AutoregressivePreprocessor,
}
