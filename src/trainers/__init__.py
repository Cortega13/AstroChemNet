"""Trainer registry."""

from src.trainers.autoencoder_trainer import AutoencoderTrainer
from src.trainers.autoregressive_trainer import AutoregressiveTrainer

TRAINER_REGISTRY = {
    "autoencoder": AutoencoderTrainer,
    "emulator": AutoregressiveTrainer,
}
