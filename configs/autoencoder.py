import os
from configs.general import GeneralConfig


class AEConfig:
    columns = GeneralConfig.species
    num_columns = len(columns)
    latents_minmax_path = os.path.join(
        GeneralConfig.working_path, "utils/latents_minmax.npy"
    )

    # Model
    input_dim = GeneralConfig.num_species  # input_dim = output_dim
    hidden_dims = (160, 80)
    latent_dim = 14

    # Hyperparameters
    lr = 1e-3
    lr_decay = 0.5
    lr_decay_patience = 12
    betas = (0.99, 0.999)
    weight_decay = 1e-4
    power_weight = 20
    conservation_weight = 1e2
    batch_size = 8 * 8192
    stagnant_epoch_patience = 20
    gradient_clipping = 2
    dropout_decay_patience = 10
    dropout_reduction_factor = 0.05
    dropout = 0.3
    noise = 0.1
    shuffle_chunk_size = 1

    # Paths
    save_model = True
    pretrained_model_path = os.path.join(
        GeneralConfig.working_path, "weights/autoencodercons.pth"
    )
    save_model_path = os.path.join(
        GeneralConfig.working_path, "weights/autoencodercons.pth"
    )
