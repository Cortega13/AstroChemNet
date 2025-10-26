"""Command-line interface for AstroChemNet."""

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from AstroChemNet import data_loading as dl
from AstroChemNet import data_processing as dp
from AstroChemNet import utils
from AstroChemNet.inference import Inference
from AstroChemNet.loss import Loss
from AstroChemNet.trainer import (
    AutoencoderTrainer,
    EmulatorTrainerSequential,
    load_objects,
)


def setup_config(cfg: DictConfig) -> None:
    """Add computed fields to config."""
    cfg.dataset.initial_abundances = np.load(cfg.dataset.initial_abundances_path)
    cfg.dataset.stoichiometric_matrix = np.load(cfg.dataset.stoichiometric_matrix_path)
    cfg.dataset.species = np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()
    cfg.dataset.num_metadata = len(cfg.dataset.metadata)
    cfg.dataset.num_phys = len(cfg.dataset.phys)
    cfg.dataset.num_species = len(cfg.dataset.species)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"


def setup_autoencoder_config(cfg: DictConfig) -> None:
    """Setup config for autoencoder training."""
    setup_config(cfg)
    cfg.model.columns = cfg.dataset.species
    cfg.model.num_columns = len(cfg.model.columns)


def setup_emulator_config(cfg: DictConfig) -> None:
    """Setup config for emulator training."""
    setup_config(cfg)
    cfg.autoencoder.columns = cfg.dataset.species
    cfg.autoencoder.num_columns = len(cfg.autoencoder.columns)
    cfg.model.columns = cfg.dataset.metadata + cfg.dataset.phys + cfg.dataset.species
    cfg.model.num_columns = len(cfg.model.columns)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train_autoencoder(cfg: DictConfig) -> None:
    """Train an autoencoder model."""
    from nn_architectures.autoencoder import Autoencoder, load_autoencoder

    OmegaConf.set_struct(cfg, False)
    setup_autoencoder_config(cfg)
    OmegaConf.set_struct(cfg, True)

    print(f"Device: {cfg.device}")
    print(f"Mode: {cfg.mode}")

    processing_functions = dp.Processing(cfg, cfg.model)

    # Preprocessing step
    if cfg.mode in ["preprocess", "both"]:
        print("\n=== Preprocessing ===")
        training_np, validation_np = dl.load_datasets(cfg, cfg.model.columns)

        processing_functions.abundances_scaling(training_np)
        processing_functions.abundances_scaling(validation_np)
        training_dataset = torch.from_numpy(training_np)
        validation_dataset = torch.from_numpy(validation_np)

        # Save preprocessed data
        torch.save(
            training_dataset,
            f"{cfg.working_path}/data/autoencoder_train_preprocessed.pt",
        )
        torch.save(
            validation_dataset,
            f"{cfg.working_path}/data/autoencoder_val_preprocessed.pt",
        )
        print("Preprocessed data saved.")

        if cfg.mode == "preprocess":
            print("Preprocessing complete. Exiting.")
            return

    # Training step
    if cfg.mode in ["train", "both"]:
        print("\n=== Training ===")

        if cfg.mode == "train":
            # Load preprocessed data
            print("Loading preprocessed data...")
            training_dataset = torch.load(
                f"{cfg.working_path}/data/autoencoder_train_preprocessed.pt"
            )
            validation_dataset = torch.load(
                f"{cfg.working_path}/data/autoencoder_val_preprocessed.pt"
            )

        training_Dataset = dl.AutoencoderDataset(training_dataset)
        validation_Dataset = dl.AutoencoderDataset(validation_dataset)

        training_dataloader = dl.tensor_to_dataloader(cfg.model, training_Dataset)
        validation_dataloader = dl.tensor_to_dataloader(cfg.model, validation_Dataset)

        autoencoder = load_autoencoder(Autoencoder, cfg, cfg.model)
        optimizer, scheduler = load_objects(autoencoder, cfg.model)
        loss_functions = Loss(processing_functions, cfg, ModelConfig=cfg.model)

        autoencoder_trainer = AutoencoderTrainer(
            cfg,
            cfg.model,
            loss_functions,
            autoencoder,
            optimizer,
            scheduler,
            training_dataloader,
            validation_dataloader,
        )

        autoencoder_trainer.train()

        # Save latents minmax after training
        total_dataset = torch.vstack((training_dataset, validation_dataset))
        inference_functions = Inference(cfg, processing_functions, autoencoder)
        processing_functions.save_latents_minmax(
            cfg.model, total_dataset, inference_functions
        )


@hydra.main(
    config_path="../../configs", config_name="config_emulator", version_base=None
)
def train_emulator(cfg: DictConfig) -> None:
    """Train an emulator model."""
    from nn_architectures.autoencoder import Autoencoder, load_autoencoder
    from nn_architectures.emulator import Emulator, load_emulator

    OmegaConf.set_struct(cfg, False)
    setup_emulator_config(cfg)
    OmegaConf.set_struct(cfg, True)

    print(f"Device: {cfg.device}")
    print(f"Mode: {cfg.mode}")

    processing_functions = dp.Processing(cfg, cfg.autoencoder)
    autoencoder = load_autoencoder(Autoencoder, cfg, cfg.autoencoder, inference=True)
    inference_functions = Inference(cfg, processing_functions, autoencoder)

    # Preprocessing step
    if cfg.mode in ["preprocess", "both"]:
        print("\n=== Preprocessing ===")
        training_np, validation_np = dl.load_datasets(cfg, cfg.model.columns)
        training_dataset = dp.preprocessing_emulator_dataset(
            cfg, cfg.model, training_np, processing_functions, inference_functions
        )
        validation_dataset = dp.preprocessing_emulator_dataset(
            cfg,
            cfg.model,
            validation_np,
            processing_functions,
            inference_functions,
        )

        dl.save_tensors_to_hdf5(cfg, training_dataset, category="training_seq")
        dl.save_tensors_to_hdf5(cfg, validation_dataset, category="validation_seq")
        print("Preprocessed data saved to HDF5.")

        if cfg.mode == "preprocess":
            print("Preprocessing complete. Exiting.")
            return

    # Training step
    if cfg.mode in ["train", "both"]:
        print("\n=== Training ===")
        print("Loading preprocessed data from HDF5...")

        training_dataset, training_indices = dl.load_tensors_from_hdf5(
            cfg, category="training_seq"
        )
        validation_dataset, validation_indices = dl.load_tensors_from_hdf5(
            cfg, category="validation_seq"
        )

        training_Dataset = dl.EmulatorSequenceDataset(
            cfg, cfg.model, training_dataset, training_indices
        )
        validation_Dataset = dl.EmulatorSequenceDataset(
            cfg, cfg.model, validation_dataset, validation_indices
        )
        del training_dataset, validation_dataset, training_indices, validation_indices

        training_dataloader = dl.tensor_to_dataloader(cfg.model, training_Dataset)
        validation_dataloader = dl.tensor_to_dataloader(cfg.model, validation_Dataset)

        emulator = load_emulator(Emulator, cfg, cfg.model)
        optimizer, scheduler = load_objects(emulator, cfg.model)

        loss_functions = Loss(
            processing_functions,
            cfg,
            ModelConfig=cfg.model,
        )
        emulator_trainer = EmulatorTrainerSequential(
            cfg,
            cfg.autoencoder,
            cfg.model,
            loss_functions,
            processing_functions,
            autoencoder,
            emulator,
            optimizer,
            scheduler,
            training_dataloader,
            validation_dataloader,
        )
        emulator_trainer.train()


def setup_preprocess_config(cfg: DictConfig) -> None:
    """Setup config for preprocessing."""
    cfg.dataset.species = np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()
    cfg.dataset.num_metadata = len(cfg.dataset.metadata)
    cfg.dataset.num_phys = len(cfg.dataset.phys)
    cfg.dataset.num_species = len(cfg.dataset.species)


@hydra.main(config_path="../../configs", config_name="preprocess", version_base=None)
def preprocess_dataset(cfg: DictConfig) -> None:
    """Preprocess raw HDF5 dataset into train/validation split."""
    OmegaConf.set_struct(cfg, False)
    setup_preprocess_config(cfg)
    OmegaConf.set_struct(cfg, True)

    print(f"Input: {cfg.input_path}")
    print(f"Output: {cfg.output_path}")
    print(f"Train/Val split: {cfg.train_split:.1%}/{1 - cfg.train_split:.1%}")
    print(f"Random seed: {cfg.seed}")

    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load initial abundances
    initial_abundances = np.load(cfg.dataset.initial_abundances_path)
    df_inits = pd.DataFrame(initial_abundances, columns=cfg.dataset.species)
    for param in ["Radfield", "Time", "Av", "gasTemp", "Density"]:
        df_inits[param] = 0

    # Load and process raw data
    print(f"\nLoading data from {cfg.input_path}...")
    df = pd.read_hdf(cfg.input_path, key=cfg.input_key, start=0)

    # Drop unwanted columns
    df = df.drop(columns=cfg.columns_to_drop)

    # Rename columns
    df.columns = utils.rename_columns(df.columns)

    # Get species columns
    species = [
        col for col in df.columns if col not in cfg.dataset.metadata + cfg.dataset.phys
    ]
    species = sorted(species)

    # Sort by model and time
    df.sort_values(by=["Model", "Time"], inplace=True)

    # Apply minimum radfield
    df["Radfield"] = np.maximum(df["Radfield"], cfg.min_radfield)

    # Process each model (tracer) separately
    print("Processing trajectories...")
    output_chunks = []
    for tracer, tdf in df.groupby("Model", sort=False):
        tdf = tdf.reset_index(drop=True)

        # Add initial conditions
        df_inits["Model"] = tdf.iloc[0]["Model"]
        tdf = pd.concat([df_inits, tdf], ignore_index=True)

        # Shift physical parameters by one timestep
        physical = tdf[cfg.params].shift(-1)
        physical.iloc[-1] = physical.iloc[-2]
        tdf[cfg.params] = physical

        output_chunks.append(tdf)

    # Combine all trajectories
    df = pd.concat(output_chunks, ignore_index=True)
    df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
    df.insert(0, "Index", range(len(df)))

    # Reorder columns
    df = df[["Index", "Model", "Time"] + cfg.params + species]

    # Split into train/validation
    print("Splitting into train/validation sets...")
    tracers = df["Model"].unique()
    np.random.shuffle(tracers)

    split_idx = int(len(tracers) * cfg.train_split)
    train_tracers = tracers[:split_idx]
    val_tracers = tracers[split_idx:]

    train_df = df[df["Model"].isin(train_tracers)].reset_index(drop=True)
    val_df = df[df["Model"].isin(val_tracers)].reset_index(drop=True)

    print(f"Train: {len(train_tracers)} models, {len(train_df)} rows")
    print(f"Val: {len(val_tracers)} models, {len(val_df)} rows")

    # Save to HDF5
    print(f"\nSaving to {cfg.output_path}...")
    train_df.to_hdf(cfg.output_path, key="train", mode="w")
    val_df.to_hdf(cfg.output_path, key="val", mode="a")

    print("Preprocessing complete!")
