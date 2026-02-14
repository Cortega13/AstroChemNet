"""CLI command handlers for training, preprocessing, and benchmarking."""

from src.configs.autoencoder import AEConfig
from src.configs.emulator import EMConfig
from src.configs.general import GeneralConfig
from src.models.autoencoder import Autoencoder
from src.models.emulator import Emulator

from . import benchmark
from .preprocessing.uclchem_grav_preprocessing import get_preprocessor
from .training.train_autoencoder import main as train_autoencoder
from .training.train_emulator import main as train_emulator


def handle_train(model: str, dataset_name: str = "uclchem_grav") -> None:
    """Handle train command for autoencoder or emulator.

    Args:
        model: Model type to train ('autoencoder' or 'emulator')
        dataset_name: Name of the dataset to use for training
    """
    general_config = GeneralConfig(dataset_name=dataset_name)

    if model == "autoencoder":
        ae_config = AEConfig(general_config=general_config)
        print(f"Training Autoencoder on {general_config.device}")
        train_autoencoder(Autoencoder, general_config, ae_config)

    elif model == "emulator":
        ae_config = AEConfig(general_config=general_config)
        em_config = EMConfig(general_config=general_config, ae_config=ae_config)
        print(f"Training Emulator on {general_config.device}")
        train_emulator(Autoencoder, Emulator, general_config, ae_config, em_config)


def handle_preprocess(dataset: str, force: bool = False) -> None:
    """Handle preprocess command for dataset preparation.

    Args:
        dataset: Dataset name to preprocess
        force: If True, overwrite existing preprocessing output
    """
    preprocessor = get_preprocessor(dataset)
    if preprocessor:
        print(f"Preprocessing dataset: {dataset}")
        preprocessor(dataset_name=dataset, force=force)
        print(f"Preprocessing complete for: {dataset}")
    else:
        print(f"Unknown dataset: {dataset}")
        print("Available datasets: uclchem_grav")


def handle_benchmark(model: str, dataset_name: str = "uclchem_grav") -> None:
    """Handle benchmark command for model evaluation.

    Args:
        model: Model type to benchmark ('autoencoder', 'emulator', or 'combined')
        dataset_name: Name of the dataset to use for benchmarking
    """
    general_config = GeneralConfig(dataset_name=dataset_name)
    ae_config = AEConfig(general_config=general_config)

    if model == "autoencoder":
        print("Benchmarking Autoencoder...")
        results = benchmark.benchmark_autoencoder(general_config, ae_config)
        print(f"Autoencoder Results: {results}")

    elif model == "emulator":
        em_config = EMConfig(general_config=general_config, ae_config=ae_config)
        print("Benchmarking Emulator...")
        results = benchmark.benchmark_emulator(general_config, ae_config, em_config)
        print(f"Emulator Results: {results}")

    elif model == "combined":
        em_config = EMConfig(general_config=general_config, ae_config=ae_config)
        print("Benchmarking Combined Pipeline (Emulator + Autoencoder)...")
        results = benchmark.benchmark_combined(general_config, ae_config, em_config)
        print(f"Combined Results: {results}")
