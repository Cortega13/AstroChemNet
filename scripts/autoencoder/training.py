import torch
import os
import numpy as np
from AstroChemNet import data_processing as dp
from AstroChemNet import data_loading as dl
from AstroChemNet.inference import Inference
from AstroChemNet.loss import Loss
import AstroChemNet.utils as utils
from AstroChemNet.trainer import (
    AutoencoderTrainer, 
    load_objects,
)
from autoencoder.config import (
    GeneralConfig,
    AEConfig,
)
from autoencoder.nn import (
    Autoencoder
)

def load_autoencoder(GeneralConfig, AEConfig, inference=False):
    autoencoder = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dims=AEConfig.hidden_dims,
        noise=AEConfig.noise,
        dropout=AEConfig.dropout,
    ).to(GeneralConfig.device)
    if os.path.exists(AEConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        autoencoder.load_state_dict(torch.load(AEConfig.pretrained_model_path))

    if inference:
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
    
    return autoencoder


if __name__ == "__main__":
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

    autoencoder = load_autoencoder(GeneralConfig, AEConfig)
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