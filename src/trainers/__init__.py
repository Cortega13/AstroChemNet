"""Trainer registry."""

from trainers.autoencoder_trainer import AutoencoderTrainer
from trainers.emulator_trainer import EmulatorTrainer

TRAINER_REGISTRY = {
    "autoencoder": AutoencoderTrainer,
    "emulator": EmulatorTrainer,
}
