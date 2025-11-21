"""Surrogate model registry."""

from surrogates.autoencoder_emulator import AutoencoderEmulatorSurrogate

SURROGATE_REGISTRY = {
    "ae_emulator_grav": AutoencoderEmulatorSurrogate,
}
