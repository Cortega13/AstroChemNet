import os
from configs.general import GeneralConfig
from configs.autoencoder import AEConfig


class EMConfig:
    columns = GeneralConfig.metadata + GeneralConfig.phys + GeneralConfig.species
    num_columns = len(columns)

    # Model
    input_dim = GeneralConfig.num_phys + AEConfig.latent_dim
    output_dim = AEConfig.latent_dim
    hidden_dim = 256
    window_size = 240

    # Hyperparameters
    lr = 1e-3
    lr_decay = 0.5
    lr_decay_patience = 5
    betas = (0.9, 0.995)
    weight_decay = 1e-3
    power_weight = 20
    conservation_weight = 5e2
    batch_size = 6 * int(512)
    stagnant_epoch_patience = 20
    gradient_clipping = 1.0
    dropout_decay_patience = 3
    dropout_reduction_factor = 0.05
    dropout = 0.0
    shuffle = True
    shuffle_chunk_size = 1

    # Paths
    save_model = False
    pretrained_model_path = os.path.join(GeneralConfig.working_path, "weights/mlp.pth")
    save_model_path = os.path.join(GeneralConfig.working_path, "weights/mlp.pth")
