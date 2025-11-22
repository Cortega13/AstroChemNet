"""Autoregressive preprocessor for emulator training data."""

import gc
from pathlib import Path

import torch
from omegaconf import DictConfig

from ..components.autoencoder import Autoencoder, load_autoencoder
from ..data_processing import Processing, load_3d_tensors
from ..surrogates.autoencoder_emulator import Inference


def preprocess_sequences(
    dataset_cfg: DictConfig,
    dataset_3d: torch.Tensor,
    processing_functions: Processing,
    inference_functions: Inference,
) -> torch.Tensor:
    """Preprocess 3D dataset: scale parameters and encode abundances with autoencoder."""
    num_phys = dataset_cfg.num_phys

    # Get shape information
    n_tracers, n_timesteps, n_features = dataset_3d.shape

    # Reshape to 2D for scaling: (N_tracers * N_timesteps, N_features)
    dataset_flat = dataset_3d.reshape(-1, n_features)

    # Extract and scale physical parameters (in-place)
    print("Scaling physical parameters...")
    phys_params = dataset_flat[:, :num_phys]
    processing_functions.physical_parameter_scaling(phys_params)

    # Extract and scale abundances (in-place)
    print("Scaling abundances...")
    abundances = dataset_flat[:, num_phys:]
    processing_functions.abundances_scaling(abundances)

    # Encode abundances to latent space
    print("Encoding to latent space...")
    latent_components = inference_functions.encode(abundances)
    latent_components = processing_functions.latent_components_scaling(
        latent_components
    ).cpu()

    # Combine physical parameters with latents
    encoded_flat = torch.cat([phys_params, latent_components], dim=1)

    # Reshape back to 3D: (N_tracers, N_timesteps, N_phys + N_latents)
    encoded_3d = encoded_flat.reshape(n_tracers, n_timesteps, -1)

    print(f"Encoded shape: {encoded_3d.shape}")

    gc.collect()
    torch.cuda.empty_cache()

    return encoded_3d


class AutoregressivePreprocessor:
    """Preprocessor for emulator training data with 3D tensor support."""

    def __init__(self, cfg: DictConfig):
        """Initialize the preprocessor."""
        self.cfg = cfg

    def run(self, output_dir: Path):
        """Load autoencoder, preprocess emulator sequences, and save 3D tensors."""
        print("=" * 80)
        print("Preprocessing Emulator Sequences")
        print("=" * 80)

        # Load pretrained autoencoder for inference
        print("\nLoading pretrained autoencoder...")
        pretrained_path = Path(self.cfg.pretrained_model_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained autoencoder not found at {pretrained_path}. "
                "Please train the autoencoder first."
            )

        processing_functions = Processing(self.cfg.data, "cpu", self.cfg.autoencoder)
        autoencoder = load_autoencoder(
            Autoencoder, self.cfg.data, self.cfg.autoencoder, inference=True
        )
        inference_functions = Inference(
            self.cfg.data, processing_functions, autoencoder
        )

        # Load 3D datasets
        print("\nLoading 3D datasets...")
        training_3d, validation_3d = load_3d_tensors(self.cfg.data)

        print("\nPreprocessing training sequences...")
        training_encoded = preprocess_sequences(
            self.cfg.data,
            training_3d,
            processing_functions,
            inference_functions,
        )

        print("\nPreprocessing validation sequences...")
        validation_encoded = preprocess_sequences(
            self.cfg.data,
            validation_3d,
            processing_functions,
            inference_functions,
        )

        # Save to PyTorch tensors
        print("\nSaving preprocessed sequences:")
        print(f"  Training shape: {training_encoded.shape}")
        print(f"  Validation shape: {validation_encoded.shape}")

        torch.save(
            training_encoded, output_dir / "autoregressive_train_preprocessed.pt"
        )
        torch.save(
            validation_encoded, output_dir / "autoregressive_val_preprocessed.pt"
        )

        print("\nPreprocessing complete!")
        del training_3d, validation_3d, training_encoded, validation_encoded
        gc.collect()
