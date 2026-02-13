"""Main entry point for training, preprocessing, and benchmarking."""

import argparse
import sys
from pathlib import Path

# Add project root to path for src package imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACNN: Autoencoder and Emulator Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train autoencoder     Train the autoencoder model
  python run.py train emulator        Train the emulator model
  python run.py preprocess gravitational_collapse  Preprocess dataset
  python run.py benchmark combined    Benchmark full pipeline
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "model",
        choices=["autoencoder", "emulator"],
        help="Model to train",
    )

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    preprocess_parser.add_argument(
        "dataset",
        choices=["gravitational_collapse"],
        help="Dataset to preprocess",
    )

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    benchmark_parser.add_argument(
        "model",
        choices=["autoencoder", "emulator", "combined"],
        help="Model to benchmark",
    )

    args = parser.parse_args()

    # Import handlers after path is set up
    from src.cli import handle_benchmark, handle_preprocess, handle_train

    if args.command == "train":
        handle_train(args.model)
    elif args.command == "preprocess":
        handle_preprocess(args.dataset)
    elif args.command == "benchmark":
        handle_benchmark(args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
