import sys
import os
import torch

import AstroChemNet.data_processing as dp
from AstroChemNet.inference import Inference
from AstroChemNet.loss import Loss
import AstroChemNet.data_loading as dl

project_root = os.path.abspath("../../")
os.chdir(project_root)
sys.path.insert(0, project_root)

from configs.general import GeneralConfig
from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from nn_architectures.autoencoder import Autoencoder, load_autoencoder
from nn_architectures.emulator import Emulator, load_emulator

autoencoder = load_autoencoder(Autoencoder, GeneralConfig, AEConfig, inference=True)
emulator = load_emulator(Emulator, GeneralConfig, EMConfig, inference=True)

processing = dp.Processing(GeneralConfig, AEConfig)
inference = Inference(GeneralConfig, processing, autoencoder, emulator)
loss = Loss(processing, GeneralConfig, AEConfig, EMConfig)

validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
    GeneralConfig, category="validation_seq"
)

validation_Dataset = dl.EmulatorSequenceDataset(
    GeneralConfig, AEConfig, validation_dataset, validation_indices
)
del validation_indices

validation_dataloader = dl.tensor_to_dataloader(EMConfig, validation_Dataset)

err = torch.zeros(
    EMConfig.window_size - 1, GeneralConfig.num_species, device=GeneralConfig.device
)
for i, (phys, latents, targets) in enumerate(validation_dataloader):
    latents = latents.to(GeneralConfig.device)
    targets = targets.to(GeneralConfig.device)
    targets = processing.inverse_abundances_scaling(targets)
    latents = processing.inverse_latent_components_scaling(latents)
    outputs = inference.emulate(phys, latents, skip_encoder=True)
    j = torch.abs(targets - outputs) / targets
    err += j.mean(dim=0)
err /= len(validation_dataloader)


print(err.mean(dim=0).max())

print(err.mean().mean())
