"""Autoregressive preprocessor for emulator training data."""

import gc
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.components.autoencoder import Autoencoder, load_autoencoder
from src.data_processing import Processing, load_3d_tensors
from src.surrogates.autoencoder_emulator import Inference


def _flatten_dataset(dataset_3d: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Flatten a 3D dataset into 2D for scaling."""
    n_tracers, n_timesteps, n_features = dataset_3d.shape
    return dataset_3d.reshape(-1, n_features), n_tracers, n_timesteps


def _scale_inputs(
    dataset_flat: torch.Tensor, num_phys: int, processing: Processing
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale physical parameters and abundances in-place."""
    phys_params = dataset_flat[:, :num_phys]
    abundances = dataset_flat[:, num_phys:]
    print("Scaling physical parameters...")
    processing.physical_parameter_scaling(phys_params)
    print("Scaling abundances...")
    processing.abundances_scaling(abundances)
    return phys_params, abundances


def _encode_latents(
    abundances: torch.Tensor, processing: Processing, inference: Inference
) -> torch.Tensor:
    """Encode abundances to scaled latent space."""
    print("Encoding to latent space...")
    latents = inference.encode(abundances)
    return processing.latents_scaling(latents).cpu()


def preprocess_sequences(
    dataset_cfg: DictConfig,
    dataset_3d: torch.Tensor,
    processing_functions: Processing,
    inference_functions: Inference,
) -> torch.Tensor:
    """Preprocess sequences into physical+latent tensors."""
    dataset_flat, n_tracers, n_timesteps = _flatten_dataset(dataset_3d)
    phys_params, abundances = _scale_inputs(
        dataset_flat, dataset_cfg.num_phys, processing_functions
    )
    latents = _encode_latents(abundances, processing_functions, inference_functions)
    encoded_flat = torch.cat([phys_params, latents], dim=1)
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

    def _load_inference(self) -> tuple[Processing, Inference]:
        """Load processing and inference objects with pretrained autoencoder."""
        print("\nLoading pretrained autoencoder...")
        pretrained_path = Path(self.cfg.pretrained_model_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained autoencoder not found at {pretrained_path}. "
                "Please train the autoencoder first."
            )
        processing = Processing(self.cfg, "cpu", self.cfg.autoencoder)
        autoencoder = load_autoencoder(
            Autoencoder, self.cfg, self.cfg.autoencoder, inference=True
        )
        inference = Inference(self.cfg, processing, autoencoder)
        return processing, inference

    def _load_datasets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load training and validation 3D tensors."""
        print("\nLoading 3D datasets...")
        return load_3d_tensors(self.cfg)

    def _save_outputs(
        self,
        output_dir: Path,
        training_encoded: torch.Tensor,
        validation_encoded: torch.Tensor,
    ) -> None:
        """Save encoded sequences to disk."""
        train_filename = getattr(
            self.cfg.output, "train_tensor", "autoregressive_train_preprocessed.pt"
        )
        val_filename = getattr(
            self.cfg.output, "val_tensor", "autoregressive_val_preprocessed.pt"
        )
        print("\nSaving preprocessed sequences:")
        print(f"  Training shape: {training_encoded.shape}")
        print(f"  Validation shape: {validation_encoded.shape}")
        torch.save(training_encoded, output_dir / train_filename)
        torch.save(validation_encoded, output_dir / val_filename)

    def run(self, output_dir: Path) -> None:
        """Preprocess emulator sequences and save tensors."""
        print("=" * 80)
        print("Preprocessing Emulator Sequences")
        print("=" * 80)
        processing, inference = self._load_inference()
        training_3d, validation_3d = self._load_datasets()
        print("\nPreprocessing training sequences...")
        training_encoded = preprocess_sequences(
            self.cfg, training_3d, processing, inference
        )
        print("\nPreprocessing validation sequences...")
        validation_encoded = preprocess_sequences(
            self.cfg, validation_3d, processing, inference
        )
        self._save_outputs(output_dir, training_encoded, validation_encoded)
        print("\nPreprocessing complete!")
        del training_3d, validation_3d, training_encoded, validation_encoded
        gc.collect()
