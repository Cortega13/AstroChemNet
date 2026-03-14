"""Training script for latent ODE model."""

from src.configs.autoencoder import AEConfig
from src.configs.datasets import DatasetConfig
from src.configs.latent_ode import LatentODEConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..models.latent_ode import LatentODE, load_latent_ode
from ..trainer import LatentODETrainerSequential, load_objects


def main(
    Autoencoder: type[Autoencoder],
    LatentODE: type[LatentODE],
    general_config: DatasetConfig,
    ae_config: AEConfig,
    ode_config: LatentODEConfig,
) -> None:
    """Train latent ODE model with given configuration."""
    training_bundle = dl.load_named_tensors_from_hdf5(
        general_config, category="training_seq", artifact_dir="latent_ode"
    )
    validation_bundle = dl.load_named_tensors_from_hdf5(
        general_config, category="validation_seq", artifact_dir="latent_ode"
    )

    training_dataset = dl.LatentODESequenceDataset(
        general_config,
        training_bundle["dataset"],
        training_bundle["indices"],
        training_bundle["delta_t"],
        ae_config.latent_dim,
    )
    validation_dataset = dl.LatentODESequenceDataset(
        general_config,
        validation_bundle["dataset"],
        validation_bundle["indices"],
        validation_bundle["delta_t"],
        ae_config.latent_dim,
    )

    training_dataloader = dl.tensor_to_dataloader(ode_config, training_dataset)
    validation_dataloader = dl.tensor_to_dataloader(ode_config, validation_dataset)

    latent_ode = load_latent_ode(LatentODE, general_config, ode_config)
    optimizer, scheduler = load_objects(latent_ode, ode_config)

    processing_functions = dp.Processing(general_config, ae_config)
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    loss_functions = Loss(
        processing_functions,
        general_config,
        ModelConfig=ode_config,
    )
    ode_trainer = LatentODETrainerSequential(
        general_config,
        ae_config,
        ode_config,
        loss_functions,
        processing_functions,
        autoencoder,
        latent_ode,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    ode_trainer.train()


if __name__ == "__main__":
    from src.configs.factory import (
        build_ae_config,
        build_dataset_config,
        build_latent_ode_config,
    )

    general_config = build_dataset_config("uclchem_grav")
    ae_config = build_ae_config(general_config)
    ode_config = build_latent_ode_config(general_config, ae_config)
    main(Autoencoder, LatentODE, general_config, ae_config, ode_config)
