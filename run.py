"""Main entry point for training, preprocessing, and benchmarking."""

import argparse
import sys
from pathlib import Path

# Add project root to path for src package imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Main CLI entry point."""
    from src.configs.datasets import AVAILABLE_DATASETS

    parser = argparse.ArgumentParser(
        description="ACNN: Autoencoder, autoregressive, and latent ODE training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train autoencoder     Train the autoencoder model
  python run.py train autoregressive  Train the abundance autoregressive model
  python run.py train latent_autoregressive        Train the latent autoregressive model
  python run.py train latent_ode      Train the latent ODE model
  python run.py preprocess uclchem_grav  Preprocess dataset
  python run.py preprocess latent_ode --dataset-name uclchem_grav
  python run.py benchmark latent_ode --dataset uclchem_grav
  python run.py benchmark combined    Benchmark full pipeline
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "model",
        choices=["autoencoder", "autoregressive", "latent_autoregressive", "latent_ode"],
        help="Model to train",
    )
    train_parser.add_argument(
        "--dataset",
        default="uclchem_grav",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use for training",
    )

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    preprocess_parser.add_argument(
        "dataset",
        choices=[*AVAILABLE_DATASETS, "autoregressive", "latent_autoregressive", "latent_ode"],
        help="Dataset to preprocess",
    )
    preprocess_parser.add_argument(
        "--dataset-name",
        default="uclchem_grav",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use when preprocessing latent autoregressive sequences",
    )
    preprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing preprocessing output",
    )

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    benchmark_parser.add_argument(
        "model",
        choices=["autoencoder", "autoregressive", "latent_autoregressive", "latent_ode", "combined"],
        help="Model to benchmark",
    )
    benchmark_parser.add_argument(
        "--dataset",
        default="uclchem_grav",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use for benchmarking",
    )

    args = parser.parse_args()

    # Import handlers after path is set up
    from src.cli import handle_benchmark, handle_preprocess, handle_train

    if args.command == "train":
        handle_train(args.model, dataset_name=args.dataset)
    elif args.command == "preprocess":
        handle_preprocess(
            args.dataset, force=args.force, dataset_name=args.dataset_name
        )
    elif args.command == "benchmark":
        handle_benchmark(args.model, dataset_name=args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
