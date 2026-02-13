"""CLI command handlers for training, preprocessing, and benchmarking."""

from src.configs.autoencoder import AEConfig
from src.configs.emulator import EMConfig
from src.configs.general import GeneralConfig
from src.models.autoencoder import Autoencoder

from . import benchmark
from .preprocessing.preprocessing import get_preprocessor
from .training.train_autoencoder import main as train_autoencoder
from .training.train_emulator import main as train_emulator


def handle_train(model: str) -> None:
    """Handle train command for autoencoder or emulator.

    Args:
        model: Model type to train ('autoencoder' or 'emulator')
    """
    general_config = GeneralConfig()

    if model == "autoencoder":
        ae_config = AEConfig()
        print(f"Training Autoencoder on {general_config.device}")
        train_autoencoder(Autoencoder, general_config, ae_config)

    elif model == "emulator":
        print(f"Training Emulator on {general_config.device}")
        train_emulator()


def handle_preprocess(dataset: str) -> None:
    """Handle preprocess command for dataset preparation.

    Args:
        dataset: Dataset name to preprocess
    """
    preprocessor = get_preprocessor(dataset)
    if preprocessor:
        print(f"Preprocessing dataset: {dataset}")
        preprocessor()
        print(f"Preprocessing complete for: {dataset}")
    else:
        print(f"Unknown dataset: {dataset}")
        print("Available datasets: gravitational_collapse")


def handle_benchmark(model: str) -> None:
    """Handle benchmark command for model evaluation.

    Args:
        model: Model type to benchmark ('autoencoder', 'emulator', or 'combined')
    """
    general_config = GeneralConfig()
    ae_config = AEConfig()

    if model == "autoencoder":
        print("Benchmarking Autoencoder...")
        results = benchmark.benchmark_autoencoder(general_config, ae_config)
        print(f"Autoencoder Results: {results}")

    elif model == "emulator":
        em_config = EMConfig()
        print("Benchmarking Emulator...")
        results = benchmark.benchmark_emulator(general_config, ae_config, em_config)
        print(f"Emulator Results: {results}")

    elif model == "combined":
        em_config = EMConfig()
        print("Benchmarking Combined Pipeline (Emulator + Autoencoder)...")
        results = benchmark.benchmark_combined(general_config, ae_config, em_config)
        print(f"Combined Results: {results}")
