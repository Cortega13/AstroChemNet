"""Training script for emulator model."""

from src.configs.autoencoder import AEConfig
from src.configs.emulator import EMConfig
from src.configs.general import GeneralConfig

from .. import data_loading as dl
from .. import data_processing as dp
from ..inference import Inference
from ..loss import Loss
from ..models.autoencoder import Autoencoder, load_autoencoder
from ..models.emulator import Emulator, load_emulator
from ..trainer import EmulatorTrainerSequential, load_objects


def main() -> None:
    """Train emulator model with given configuration."""
    general_config = GeneralConfig()
    ae_config = AEConfig()
    em_config = EMConfig()

    print(general_config.device)
    processing_functions = dp.Processing(
        general_config,
        ae_config,
    )
    autoencoder = load_autoencoder(
        Autoencoder, general_config, ae_config, inference=True
    )
    inference_functions = Inference(general_config, processing_functions, autoencoder)

    training_np, validation_np = dl.load_datasets(general_config, em_config.columns)
    training_dataset = dp.preprocessing_emulator_dataset(
        general_config,
        em_config,
        training_np,
        processing_functions,
        inference_functions,
    )
    validation_dataset = dp.preprocessing_emulator_dataset(
        general_config,
        em_config,
        validation_np,
        processing_functions,
        inference_functions,
    )

    dl.save_tensors_to_hdf5(general_config, training_dataset, category="training_seq")
    dl.save_tensors_to_hdf5(
        general_config, validation_dataset, category="validation_seq"
    )

    # training_dataset, training_indices = dl.load_tensors_from_hdf5(
    #     general_config, category="training_seq"
    # )
    # validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
    #     general_config, category="validation_seq"
    # )

    # training_Dataset = dl.EmulatorSequenceDataset(
    #     general_config, ae_config, training_dataset, training_indices
    # )
    # validation_Dataset = dl.EmulatorSequenceDataset(
    #     general_config, ae_config, validation_dataset, validation_indices
    # )
    # del training_dataset, validation_dataset, training_indices, validation_indices

    # training_dataloader = dl.tensor_to_dataloader(em_config, training_Dataset)
    # validation_dataloader = dl.tensor_to_dataloader(em_config, validation_Dataset)

    # emulator = load_emulator(Emulator, general_config, em_config)
    # optimizer, scheduler = load_objects(emulator, em_config)

    # loss_functions = Loss(
    #     processing_functions,
    #     general_config,
    #     ModelConfig=em_config,
    # )
    # emulator_trainer = EmulatorTrainerSequential(
    #     general_config,
    #     ae_config,
    #     em_config,
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
    # Run main script.
    main()
