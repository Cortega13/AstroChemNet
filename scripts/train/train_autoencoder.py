"""Script to train an autoencoder."""

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from AstroChemNet import data_loading as dl
from AstroChemNet import data_processing as dp
from AstroChemNet.config_schemas import Config
from AstroChemNet.inference import Inference
from AstroChemNet.loss import Loss
from AstroChemNet.trainer import AutoencoderTrainer, load_objects
from nn_architectures.autoencoder import Autoencoder, load_autoencoder


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

    # Compute model columns based on model type
    if cfg.model.model_name == "autoencoder":
        cfg.model.columns = cfg.dataset.species
    elif cfg.model.model_name == "emulator":
        cfg.model.columns = (
            cfg.dataset.metadata + cfg.dataset.phys + cfg.dataset.species
        )
    else:
        cfg.model.columns = []

    cfg.model.num_columns = len(cfg.model.columns)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function for autoencoder.

    Args:
        cfg: Hydra configuration object
    """
    OmegaConf.set_struct(cfg, False)
    setup_config(cfg)
    OmegaConf.set_struct(cfg, True)

    print(f"Device: {cfg.device}")

    processing_functions = dp.Processing(cfg, cfg.model)

    training_np, validation_np = dl.load_datasets(cfg, cfg.model.columns)

    processing_functions.abundances_scaling(training_np)
    processing_functions.abundances_scaling(validation_np)
    training_dataset = torch.from_numpy(training_np)
    validation_dataset = torch.from_numpy(validation_np)

    training_Dataset = dl.AutoencoderDataset(training_dataset)
    validation_Dataset = dl.AutoencoderDataset(validation_dataset)

    training_dataloader = dl.tensor_to_dataloader(cfg.model, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(cfg.model, validation_Dataset)

    autoencoder = load_autoencoder(Autoencoder, cfg, cfg.model)
    optimizer, scheduler = load_objects(autoencoder, cfg.model)
    loss_functions = Loss(processing_functions, cfg, ModelConfig=cfg.model)

    autoencoder_trainer = AutoencoderTrainer(
        cfg,
        cfg.model,
        loss_functions,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )

    autoencoder_trainer.train()

    total_dataset = torch.vstack((training_dataset, validation_dataset))
    inference_functions = Inference(cfg, processing_functions, autoencoder)
    processing_functions.save_latents_minmax(
        cfg.model, total_dataset, inference_functions
    )


if __name__ == "__main__":
    main()
