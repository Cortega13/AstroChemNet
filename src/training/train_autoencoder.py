"""Training script for autoencoder model."""

import torch

from src.configs.autoencoder import AEConfig
from src.configs.general import GeneralConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..inference import Inference
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..trainer import AutoencoderTrainer, load_objects


def main(
    Autoencoder: type[Autoencoder],
    general_config: GeneralConfig,
    ae_config: AEConfig,
) -> None:
    """Train autoencoder model with given configuration."""
    processing_functions = dp.Processing(general_config, ae_config)

    # stoichiometric_matrix = processing_functions.save_stoichiometric_matrix()
    # print(f"Stochiometry Matrix: {stoichiometric_matrix} | Shape: {stoichiometric_matrix.shape}")

    training_np, validation_np = dl.load_datasets(general_config, ae_config.columns)

    processing_functions.abundances_scaling(training_np)
    processing_functions.abundances_scaling(validation_np)
    training_dataset = torch.from_numpy(training_np)
    validation_dataset = torch.from_numpy(validation_np)

    training_Dataset = dl.AutoencoderDataset(training_dataset)
    validation_Dataset = dl.AutoencoderDataset(validation_dataset)

    training_dataloader = dl.tensor_to_dataloader(ae_config, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(ae_config, validation_Dataset)

    autoencoder = load_autoencoder(Autoencoder, general_config, ae_config)
    optimizer, scheduler = load_objects(autoencoder, ae_config)
    loss_functions = Loss(
        processing_functions,
        general_config,
        ModelConfig=ae_config,
    )

    autoencoder_trainer = AutoencoderTrainer(
        general_config,
        ae_config,
        loss_functions,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )

    autoencoder_trainer.train()

    total_dataset = torch.vstack((training_dataset, validation_dataset))
    inference_functions = Inference(general_config, processing_functions, autoencoder)
    processing_functions.save_latents_minmax(
        ae_config, total_dataset, inference_functions
    )


if __name__ == "__main__":
    print(f"Device: {GeneralConfig.device}")
    # Run main script.
    main(Autoencoder, GeneralConfig(), AEConfig())
