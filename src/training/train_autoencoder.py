"""Training script for autoencoder model."""

import numpy as np
import torch

from src.configs.autoencoder import AEConfig
from src.configs.general import GeneralConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..inference import Inference
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..trainer import AutoencoderTrainer, load_objects


def save_latents_minmax(
    general_config: GeneralConfig,
    dataset_t: torch.Tensor,
    inference_functions: Inference,
) -> None:
    """Compute and save min/max values of latent components for scaling."""
    min_, max_ = float("inf"), float("-inf")

    with torch.no_grad():
        for i in range(0, len(dataset_t), AEConfig.batch_size):
            batch = dataset_t[i : i + AEConfig.batch_size].to(general_config.device)
            encoded = inference_functions.encode(batch).cpu()
            min_ = min(min_, encoded.min().item())
            max_ = max(max_, encoded.max().item())

    minmax_np = np.array([min_, max_], dtype=np.float32)
    print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
    np.save(AEConfig.latents_minmax_path, minmax_np)


def main(
    Autoencoder: type[Autoencoder],
    general_config: GeneralConfig,
    ae_config: AEConfig,
) -> None:
    """Train autoencoder model with given configuration."""
    processing_functions = dp.Processing(general_config)

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
    save_latents_minmax(general_config, total_dataset, inference_functions)


if __name__ == "__main__":
    # Instantiate configs - GeneralConfig loads from preprocessing output
    general_config = GeneralConfig(dataset_name="uclchem_grav")
    ae_config = AEConfig(general_config=general_config)

    print(f"Device: {general_config.device}")
    # Run main script.
    main(Autoencoder, general_config, ae_config)
