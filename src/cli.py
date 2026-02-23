"""CLI command handlers for training, preprocessing, and benchmarking."""

from src.configs.factory import build_ae_config, build_dataset_config, build_em_config
from src.models.autoencoder import Autoencoder
from src.models.emulator import Emulator

from . import benchmark
from .preprocessing import (
    get_preprocessor,
    list_available_datasets,
    preprocess_emulator,
)
from .training.train_autoencoder import main as train_autoencoder
from .training.train_emulator import main as train_emulator


def handle_train(model: str, dataset_name: str = "uclchem_grav") -> None:
    """Handle train command for autoencoder or emulator.

    Args:
        model: Model type to train ('autoencoder' or 'emulator')
        dataset_name: Name of the dataset to use for training
    """
    dataset_config = build_dataset_config(dataset_name)

    if model == "autoencoder":
        ae_config = build_ae_config(dataset_config)
        print(f"Training Autoencoder on {dataset_config.device}")
        train_autoencoder(Autoencoder, dataset_config, ae_config)

    elif model == "emulator":
        ae_config = build_ae_config(dataset_config)
        em_config = build_em_config(dataset_config, ae_config)
        print(f"Training Emulator on {dataset_config.device}")
        train_emulator(Autoencoder, Emulator, dataset_config, ae_config, em_config)


def handle_preprocess(
    dataset: str,
    force: bool = False,
    dataset_name: str = "uclchem_grav",
) -> None:
    """Handle preprocess command for dataset preparation.

    Args:
        dataset: Dataset name to preprocess ('uclchem_grav' or 'emulator')
        force: If True, overwrite existing preprocessing output
    """
    # Special handling for emulator preprocessing
    if dataset == "emulator":
        dataset_config = build_dataset_config(dataset_name)
        ae_config = build_ae_config(dataset_config)
        em_config = build_em_config(dataset_config, ae_config)
        print("Preprocessing emulator dataset...")
        preprocess_emulator(dataset_config, ae_config, em_config, Autoencoder)
        print("Emulator preprocessing complete.")
        return

    # Standard dataset preprocessing
    preprocessor = get_preprocessor(dataset)
    if preprocessor:
        print(f"Preprocessing dataset: {dataset}")
        preprocessor(dataset_name=dataset, force=force)
        print(f"Preprocessing complete for: {dataset}")
    else:
        print(f"Unknown dataset: {dataset}")
        print(f"Available datasets: {', '.join(list_available_datasets())}")


def handle_benchmark(model: str, dataset_name: str = "uclchem_grav") -> None:
    """Handle benchmark command for model evaluation.

    Args:
        model: Model type to benchmark ('autoencoder', 'emulator', or 'combined')
        dataset_name: Name of the dataset to use for benchmarking
    """
    dataset_config = build_dataset_config(dataset_name)
    ae_config = build_ae_config(dataset_config)

    if model == "autoencoder":
        print("Benchmarking Autoencoder...")
        results = benchmark.benchmark_autoencoder(dataset_config, ae_config)
        print(f"Autoencoder Results: {results}")

    elif model == "emulator":
        em_config = build_em_config(dataset_config, ae_config)
        print("Benchmarking Emulator...")
        results = benchmark.benchmark_emulator(dataset_config, ae_config, em_config)
        print(f"Emulator Results: {results}")

    elif model == "combined":
        em_config = build_em_config(dataset_config, ae_config)
        print("Benchmarking Combined Pipeline (Emulator + Autoencoder)...")
        results = benchmark.benchmark_combined(dataset_config, ae_config, em_config)
        print(f"Combined Results: {results}")
