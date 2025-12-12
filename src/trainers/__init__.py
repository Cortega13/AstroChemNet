"""Registers available trainer classes."""

from src.trainers.autoencoder_trainer import AutoencoderTrainer
from src.trainers.emulator_trainer import EmulatorTrainer

TRAINER_REGISTRY = {
    "autoencoder": AutoencoderTrainer,
    "emulator": EmulatorTrainer,
}
