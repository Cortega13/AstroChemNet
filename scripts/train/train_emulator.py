"""Script to train an emulator."""

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from AstroChemNet import data_loading as dl
from AstroChemNet import data_processing as dp
from AstroChemNet.config_schemas import Config
from AstroChemNet.inference import Inference
from AstroChemNet.loss import Loss
from AstroChemNet.trainer import EmulatorTrainerSequential, load_objects
from nn_architectures.autoencoder import Autoencoder, load_autoencoder
from nn_architectures.emulator import Emulator, load_emulator


def setup_config(cfg: DictConfig) -> None:
    """Add computed fields to config."""
    cfg.dataset.initial_abundances = np.load(cfg.dataset.initial_abundances_path)
    cfg.dataset.stoichiometric_matrix = np.load(cfg.dataset.stoichiometric_matrix_path)
    cfg.dataset.species = np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()
    cfg.dataset.num_metadata = len(cfg.dataset.metadata)
    cfg.dataset.num_phys = len(cfg.dataset.phys)
    cfg.dataset.num_species = len(cfg.dataset.species)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    # Compute autoencoder columns
    cfg.autoencoder.columns = cfg.dataset.species
    cfg.autoencoder.num_columns = len(cfg.autoencoder.columns)

    # Compute emulator columns
    cfg.model.columns = (
        cfg.dataset.metadata + cfg.dataset.phys + cfg.dataset.species
    )
    cfg.model.num_columns = len(cfg.model.columns)


@hydra.main(config_path="../../configs", config_name="config_emulator", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function for emulator.

    Args:
        cfg: Hydra configuration object
    """
    OmegaConf.set_struct(cfg, False)
    setup_config(cfg)
    OmegaConf.set_struct(cfg, True)

    print(f"Device: {cfg.device}")

    processing_functions = dp.Processing(cfg, cfg.autoencoder)
    autoencoder = load_autoencoder(Autoencoder, cfg, cfg.autoencoder, inference=True)
    inference_functions = Inference(cfg, processing_functions, autoencoder)

    training_np, validation_np = dl.load_datasets(cfg, cfg.model.columns)
    training_dataset = dp.preprocessing_emulator_dataset(
        cfg, cfg.model, training_np, processing_functions, inference_functions
    )
    validation_dataset = dp.preprocessing_emulator_dataset(
        cfg,
        cfg.model,
        validation_np,
        processing_functions,
        inference_functions,
    )

    dl.save_tensors_to_hdf5(cfg, training_dataset, category="training_seq")
    dl.save_tensors_to_hdf5(cfg, validation_dataset, category="validation_seq")

    # training_dataset, training_indices = dl.load_tensors_from_hdf5(
    #     cfg, category="training_seq"
    # )
    # validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
    #     cfg, category="validation_seq"
    # )

    # training_Dataset = dl.EmulatorSequenceDataset(
    #     cfg, cfg.model, training_dataset, training_indices
    # )
    # validation_Dataset = dl.EmulatorSequenceDataset(
    #     cfg, cfg.model, validation_dataset, validation_indices
    # )
    # del training_dataset, validation_dataset, training_indices, validation_indices

    # training_dataloader = dl.tensor_to_dataloader(cfg.model, training_Dataset)
    # validation_dataloader = dl.tensor_to_dataloader(cfg.model, validation_Dataset)

    # emulator = load_emulator(Emulator, cfg, cfg.model)
    # optimizer, scheduler = load_objects(emulator, cfg.model)

    # loss_functions = Loss(
    #     processing_functions,
    #     cfg,
    #     ModelConfig=cfg.model,
    # )
    # emulator_trainer = EmulatorTrainerSequential(
    #     cfg,
    #     cfg.autoencoder,
    #     cfg.model,
    #     loss_functions,
    #     processing_functions,
    #     autoencoder,
    #     emulator,
    #     optimizer,
    #     scheduler,
    #     training_dataloader,
    #     validation_dataloader,
    # )
    # emulator_trainer.train()


if __name__ == "__main__":
    main()
