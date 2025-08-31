import os
import torch
import numpy as np
from AstroChemNet import data_processing as dp
from AstroChemNet import data_loading as dl
from AstroChemNet.loss import Loss
from AstroChemNet.inference import Inference
import AstroChemNet.utils as utils
from AstroChemNet.trainer import (
    AutoencoderTrainer, 
    load_objects,
)
os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
from configs.general import GeneralConfig
from configs.autoencoder import AEConfig
from nn_architectures.autoencoder import load_autoencoder, Autoencoder


def main(Autoencoder, GeneralConfig, AEConfig):
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
        AEConfig=AEConfig,
    )
    
    autoencoder_trainer = AutoencoderTrainer(
        GeneralConfig,
        AEConfig,
        loss_functions,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
    )
    
    autoencoder_trainer.train()
    
    # total_dataset = torch.vstack((training_dataset, validation_dataset))
    # inference_functions = Inference(GeneralConfig, processing_functions, autoencoder)
    # processing_functions.save_latents_minmax(AEConfig, total_dataset, inference_functions)
    

if __name__ == "__main__":
    # Run main script.
    main(Autoencoder, GeneralConfig, AEConfig)