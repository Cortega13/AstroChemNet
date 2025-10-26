import os
import sys

import AstroChemNet.data_loading as dl
import AstroChemNet.data_processing as dp
from AstroChemNet.inference import Inference

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(project_root)
sys.path.insert(0, project_root)

from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from configs.general import GeneralConfig
from nn_architectures.autoencoder import Autoencoder, load_autoencoder

autoencoder = load_autoencoder(Autoencoder, GeneralConfig, AEConfig, inference=True)

processing = dp.Processing(GeneralConfig, AEConfig)
inference = Inference(GeneralConfig, processing, autoencoder)


training_np, validation_np = dl.load_datasets(GeneralConfig, EMConfig.columns)
del validation_np
