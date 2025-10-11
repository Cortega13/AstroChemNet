"""Script to train an autoencoder."""

import torch

from AstroChemNet import data_loading as dl
from AstroChemNet import data_processing as dp
from AstroChemNet.inference import Inference
from AstroChemNet.loss import Loss
from AstroChemNet.trainer import AutoencoderTrainer, load_objects
from configs.autoencoder import AEConfig
from configs.general import GeneralConfig
from nn_architectures.autoencoder import Autoencoder, load_autoencoder


def main(
    Autoencoder: type[Autoencoder],
    GeneralConfig: type[GeneralConfig],
    AEConfig: type[AEConfig],
):
    """Main _summary_.

    Args:
        Autoencoder (type[Autoencoder]): nn.module of Autoencoder
        GeneralConfig (type[GeneralConfig]): ConfigurationFile
        AEConfig (type[AEConfig]): Autoencoder-specific configuration file.
    """
    processing_functions = dp.Processing(GeneralConfig, AEConfig)

    # stoichiometric_matrix = processing_functions.save_stoichiometric_matrix()
    # print(f"Stochiometry Matrix: {stoichiometric_matrix} | Shape: {stoichiometric_matrix.shape}")

    training_np, validation_np = dl.load_datasets(GeneralConfig, AEConfig.columns)

    processing_functions.abundances_scaling(training_np)
    processing_functions.abundances_scaling(validation_np)
    training_dataset = torch.from_numpy(training_np)
    validation_dataset = torch.from_numpy(validation_np)

    training_Dataset = dl.AutoencoderDataset(training_dataset)
    validation_Dataset = dl.AutoencoderDataset(validation_dataset)

    training_dataloader = dl.tensor_to_dataloader(AEConfig, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(AEConfig, validation_Dataset)

    autoencoder = load_autoencoder(Autoencoder, GeneralConfig, AEConfig)
    optimizer, scheduler = load_objects(autoencoder, AEConfig)
    loss_functions = Loss(
        processing_functions,
        GeneralConfig,
        ModelConfig=AEConfig,
    )

    autoencoder_trainer = AutoencoderTrainer(
        GeneralConfig,
        AEConfig,
        loss_functions,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )

    autoencoder_trainer.train()

    total_dataset = torch.vstack((training_dataset, validation_dataset))
    inference_functions = Inference(GeneralConfig, processing_functions, autoencoder)
    processing_functions.save_latents_minmax(
        AEConfig, total_dataset, inference_functions
    )


if __name__ == "__main__":
    print(f"Device: {GeneralConfig.device}")
    # Run main script.
    main(Autoencoder, GeneralConfig, AEConfig)
