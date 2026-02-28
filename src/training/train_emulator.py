"""Training script for emulator model."""

from src.configs.autoencoder import AEConfig
from src.configs.datasets import DatasetConfig
from src.configs.emulator import EMConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..models.emulator import Emulator, load_emulator
from ..trainer import EmulatorTrainerSequential, load_objects

# Optional PyTorch profiler (writes a Chrome trace JSON).
PROFILE_TRAINING = False
PROFILE_TRACE_PATH = "outputs/emulator_training_trace.json"


def main(
    Autoencoder: type[Autoencoder],
    Emulator: type[Emulator],
    general_config: DatasetConfig,
    ae_config: AEConfig,
    em_config: EMConfig,
) -> None:
    """Train emulator model with given configuration.

    Note: Run 'python run.py preprocess emulator' before training to prepare
    the training and validation datasets.
    """
    # Load preprocessed datasets (must run preprocessing first)
    training_dataset, training_indices = dl.load_tensors_from_hdf5(
        general_config, category="training_seq"
    )
    validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
        general_config, category="validation_seq"
    )

    training_Dataset = dl.EmulatorSequenceDataset(
        general_config, ae_config, training_dataset, training_indices
    )
    validation_Dataset = dl.EmulatorSequenceDataset(
        general_config, ae_config, validation_dataset, validation_indices
    )
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dl.tensor_to_dataloader(em_config, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(em_config, validation_Dataset)

    emulator = load_emulator(Emulator, general_config, em_config)
    optimizer, scheduler = load_objects(emulator, em_config)

    processing_functions = dp.Processing(general_config, ae_config)
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )

    loss_functions = Loss(
        processing_functions,
        general_config,
        ModelConfig=em_config,
    )
    emulator_trainer = EmulatorTrainerSequential(
        general_config,
        ae_config,
        em_config,
        loss_functions,
        processing_functions,
        autoencoder,
        emulator,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    if PROFILE_TRAINING:
        import torch

        acts = [torch.profiler.ProfilerActivity.CPU] + (
            [torch.profiler.ProfilerActivity.CUDA]
            if str(general_config.device) == "cuda"
            else []
        )
        with torch.profiler.profile(activities=acts, record_shapes=True) as p:
            emulator_trainer.train()
        p.export_chrome_trace(PROFILE_TRACE_PATH)
    else:
        emulator_trainer.train()


if __name__ == "__main__":
    # Instantiate configs - GeneralConfig loads from preprocessing output
    from src.configs.factory import (
        build_ae_config,
        build_dataset_config,
        build_em_config,
    )

    general_config = build_dataset_config("uclchem_grav")
    ae_config = build_ae_config(general_config)
    em_config = build_em_config(general_config, ae_config)

    # Run main script.
    main(Autoencoder, Emulator, general_config, ae_config, em_config)
