import os
from AstroChemNet import data_processing as dp
from AstroChemNet.inference import Inference  # noqa: F401
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(project_root)
sys.path.insert(0, project_root)
from configs.general import GeneralConfig
from configs.autoencoder import AEConfig
from configs.emulator import EMConfig  # noqa: F401
from nn_architectures.autoencoder import Autoencoder, load_autoencoder  # noqa: F401
from nn_architectures.emulator import Emulator, load_emulator  # noqa: F401
from AstroChemNet import data_loading as dl  # noqa: F401
from AstroChemNet.trainer import EmulatorTrainerSequential, load_objects  # noqa: F401
from AstroChemNet.loss import Loss  # noqa: F401


def main():
    processing_functions = dp.Processing(
        GeneralConfig,
        AEConfig,
    )
    autoencoder = load_autoencoder(Autoencoder, GeneralConfig, AEConfig, inference=True)
    inference_functions = Inference(GeneralConfig, processing_functions, autoencoder)

    training_np, validation_np = dl.load_datasets(GeneralConfig, EMConfig.columns)
    training_dataset = dp.preprocessing_emulator_dataset(
        GeneralConfig, EMConfig, training_np, processing_functions, inference_functions
    )
    validation_dataset = dp.preprocessing_emulator_dataset(
        GeneralConfig,
        EMConfig,
        validation_np,
        processing_functions,
        inference_functions,
    )

    dl.save_tensors_to_hdf5(GeneralConfig, training_dataset, category="training_seq")
    dl.save_tensors_to_hdf5(
        GeneralConfig, validation_dataset, category="validation_seq"
    )

    # training_dataset, training_indices = dl.load_tensors_from_hdf5(
    #     GeneralConfig, category="training_seq"
    # )
    # validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
    #     GeneralConfig, category="validation_seq"
    # )

    # training_Dataset = dl.EmulatorSequenceDataset(
    #     GeneralConfig, AEConfig, training_dataset, training_indices
    # )
    # validation_Dataset = dl.EmulatorSequenceDataset(
    #     GeneralConfig, AEConfig, validation_dataset, validation_indices
    # )
    # del training_dataset, validation_dataset, training_indices, validation_indices

    # training_dataloader = dl.tensor_to_dataloader(EMConfig, training_Dataset)
    # validation_dataloader = dl.tensor_to_dataloader(EMConfig, validation_Dataset)

    # emulator = load_emulator(Emulator, GeneralConfig, EMConfig)
    # optimizer, scheduler = load_objects(emulator, EMConfig)

    # loss_functions = Loss(
    #     processing_functions,
    #     GeneralConfig,
    #     ModelConfig=EMConfig,
    # )
    # emulator_trainer = EmulatorTrainerSequential(
    #     GeneralConfig,
    #     AEConfig,
    #     EMConfig,
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
