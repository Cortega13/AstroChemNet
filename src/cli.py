"""CLI command handlers for training, preprocessing, and benchmarking."""

from src.configs.factory import (
    build_ae_config,
    build_ar_config,
    build_autoregressive_config,
    build_dataset_config,
    build_latent_ode_config,
)
from src.models.autoencoder import Autoencoder
from src.models.autoregressive import Autoregressive
from src.models.latent_autoregressive import LatentAR
from src.models.latent_ode import LatentODE

from . import benchmark
from .preprocessing import (
    get_preprocessor,
    list_available_datasets,
    preprocess_autoregressive,
    preprocess_latent_autoregressive,
    preprocess_latent_ode,
)
from .training.train_autoencoder import main as train_autoencoder
from .training.train_autoregressive import main as train_autoregressive
from .training.train_latent_autoregressive import main as train_latent_autoregressive
from .training.train_latent_ode import main as train_latent_ode


def handle_train(model: str, dataset_name: str = "uclchem_grav") -> None:
    """Handle train command for autoencoder or latent autoregressive.

    Args:
        model: Model type to train ('autoencoder', 'autoregressive', or 'latent_autoregressive')
        dataset_name: Name of the dataset to use for training
    """
    dataset_config = build_dataset_config(dataset_name)

    if model == "autoencoder":
        ae_config = build_ae_config(dataset_config)
        print(f"Training Autoencoder on {dataset_config.device}")
        train_autoencoder(Autoencoder, dataset_config, ae_config)

    elif model == "latent_autoregressive":
        ae_config = build_ae_config(dataset_config)
        ar_config = build_ar_config(dataset_config, ae_config)
        print(f"Training LatentAR on {dataset_config.device}")
        train_latent_autoregressive(
            Autoencoder, LatentAR, dataset_config, ae_config, ar_config
        )

    elif model == "latent_ode":
        ae_config = build_ae_config(dataset_config)
        ode_config = build_latent_ode_config(dataset_config, ae_config)
        print(f"Training LatentODE on {dataset_config.device}")
        train_latent_ode(
            Autoencoder, LatentODE, dataset_config, ae_config, ode_config
        )

    elif model == "autoregressive":
        ar_config = build_autoregressive_config(dataset_config)
        print(f"Training Autoregressive on {dataset_config.device}")
        train_autoregressive(Autoregressive, dataset_config, ar_config)


def handle_preprocess(
    dataset: str,
    force: bool = False,
    dataset_name: str = "uclchem_grav",
) -> None:
    """Handle preprocess command for dataset preparation.

    Args:
        dataset: Dataset name to preprocess ('uclchem_grav', 'autoregressive', or 'latent_autoregressive')
        force: If True, overwrite existing preprocessing output
        dataset_name: Dataset to use when building autoregressive sequence artifacts
    """
    if dataset == "autoregressive":
        dataset_config = build_dataset_config(dataset_name)
        ar_config = build_autoregressive_config(dataset_config)
        print("Preprocessing autoregressive dataset...")
        preprocess_autoregressive(dataset_config, ar_config)
        print("Autoregressive preprocessing complete.")
        return

    # Special handling for latent autoregressive preprocessing
    if dataset == "latent_autoregressive":
        dataset_config = build_dataset_config(dataset_name)
        ae_config = build_ae_config(dataset_config)
        ar_config = build_ar_config(dataset_config, ae_config)
        print("Preprocessing latent autoregressive dataset...")
        preprocess_latent_autoregressive(
            dataset_config, ae_config, ar_config, Autoencoder
        )
        print("LatentAR preprocessing complete.")
        return

    if dataset == "latent_ode":
        dataset_config = build_dataset_config(dataset_name)
        ae_config = build_ae_config(dataset_config)
        ode_config = build_latent_ode_config(dataset_config, ae_config)
        print("Preprocessing latent ODE dataset...")
        preprocess_latent_ode(dataset_config, ae_config, ode_config, Autoencoder)
        print("LatentODE preprocessing complete.")
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
        model: Model type to benchmark ('autoencoder', 'autoregressive', 'latent_autoregressive', or 'combined')
        dataset_name: Name of the dataset to use for benchmarking
    """
    dataset_config = build_dataset_config(dataset_name)
    ae_config = build_ae_config(dataset_config)

    if model == "autoencoder":
        print("Benchmarking Autoencoder...")
        results = benchmark.benchmark_autoencoder(dataset_config, ae_config)
        print(f"Autoencoder Results: {results}")

    elif model == "latent_autoregressive":
        ar_config = build_ar_config(dataset_config, ae_config)
        print("Benchmarking LatentAR...")
        results = benchmark.benchmark_latent_autoregressive(
            dataset_config, ae_config, ar_config
        )
        print(f"LatentAR Results: {results}")

    elif model == "autoregressive":
        ar_config = build_autoregressive_config(dataset_config)
        print("Benchmarking Autoregressive...")
        results = benchmark.benchmark_autoregressive(dataset_config, ar_config)
        print(f"Autoregressive Results: {results}")

    elif model == "latent_ode":
        ode_config = build_latent_ode_config(dataset_config, ae_config)
        print("Benchmarking LatentODE...")
        results = benchmark.benchmark_latent_ode(
            dataset_config, ae_config, ode_config
        )
        print(f"LatentODE Results: {results}")

    elif model == "combined":
        ar_config = build_ar_config(dataset_config, ae_config)
        print("Benchmarking Combined Pipeline (LatentAR + Autoencoder)...")
        results = benchmark.benchmark_combined(dataset_config, ae_config, ar_config)
        print(f"Combined Results: {results}")
