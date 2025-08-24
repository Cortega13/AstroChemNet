import torch
import os

from AstroChemNet import data_processing as dp
from AstroChemNet import data_loading as dl
from AstroChemNet.loss import Loss
from AstroChemNet.inference import (
    Inference
)
from AstroChemNet.trainer import (
    EmulatorTrainerSequential,
    load_objects
)
from emulator.config import (
    GeneralConfig,
    AEConfig,
    EMConfig
)
from emulator.nn import (
    Autoencoder,
    Emulator
)

def load_autoencoder(GeneralConfig, AEConfig):
    autoencoder = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dims=AEConfig.hidden_dims,
    ).to(GeneralConfig.device)
    if os.path.exists(AEConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        autoencoder.load_state_dict(torch.load(AEConfig.pretrained_model_path))
    
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    return autoencoder


def load_emulator(GeneralConfig, EMConfig, inference=False):
    emulator = Emulator(
        input_dim=EMConfig.input_dim,
        output_dim=EMConfig.output_dim,
        hidden_dim=EMConfig.hidden_dim
    ).to(GeneralConfig.device)
    if os.path.exists(EMConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        emulator.load_state_dict(torch.load(EMConfig.pretrained_model_path))
    if inference:
        emulator.eval()
        for param in emulator.parameters():
            param.requires_grad = False
    return emulator


if __name__ == "__main__":
    processing_functions = dp.Processing(
        GeneralConfig, 
        AEConfig,
    )
    autoencoder = load_autoencoder(GeneralConfig, AEConfig)
    inference_functions = Inference(GeneralConfig, processing_functions, autoencoder)

    # training_np, validation_np = dl.load_datasets(GeneralConfig, EMConfig.columns)
    # training_dataset = dp.preprocessing_emulator_dataset(GeneralConfig, EMConfig, training_np, processing_functions, inference_functions)
    # validation_dataset = dp.preprocessing_emulator_dataset(GeneralConfig, EMConfig, validation_np, processing_functions, inference_functions)
    
    # dl.save_tensors_to_hdf5(GeneralConfig, training_dataset, category="training_seq")
    # dl.save_tensors_to_hdf5(GeneralConfig, validation_dataset, category="validation_seq")
    
    training_dataset, training_indices = dl.load_tensors_from_hdf5(GeneralConfig, category="training_seq")
    validation_dataset, validation_indices = dl.load_tensors_from_hdf5(GeneralConfig, category="validation_seq")
    
    training_Dataset = dl.EmulatorSequenceDataset(GeneralConfig, AEConfig, training_dataset, training_indices)
    validation_Dataset = dl.EmulatorSequenceDataset(GeneralConfig, AEConfig, validation_dataset, validation_indices)
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dl.tensor_to_dataloader(EMConfig, training_Dataset)
    validation_dataloader = dl.tensor_to_dataloader(EMConfig, validation_Dataset)
    
    
    emulator = load_emulator(GeneralConfig, EMConfig)
    optimizer, scheduler = load_objects(emulator, EMConfig)
    
    loss_functions = Loss(
        processing_functions,
        GeneralConfig,
        EMConfig=EMConfig,
    )
    emulator_trainer = EmulatorTrainerSequential(
        GeneralConfig,
        AEConfig,
        EMConfig,
        loss_functions,
        processing_functions,
        autoencoder,
        emulator,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    emulator_trainer.train()