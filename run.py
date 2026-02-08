"""Runs train, preprocess, or benchmark from one entrypoint; example: `python run.py preprocess uclchem_grav`; assumptions: config registries in `src/configs` define valid dataset/component/surrogate names."""

import argparse

from src.benchmark import run_benchmark
from src.preprocess import run_preprocess
from src.train import run_training


def _build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser with train/preprocess/benchmark subcommands."""
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "component", nargs="?", default="autoencoder_uclchem_grav"
    )

    prep_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    prep_parser.add_argument(
        "target",
        nargs="?",
        default="uclchem_grav",
        help=(
            "One dataset/component target. "
            "Examples: preprocess uclchem_grav, preprocess autoencoder_uclchem_grav"
        ),
    )

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarking")
    bench_parser.add_argument(
        "surrogate", nargs="?", default="autoencoder_emulator_uclchem_grav"
    )
    return parser


def main() -> None:
    """Dispatches the selected subcommand."""
    args = _build_parser().parse_args()
    if args.command == "train":
        run_training(args.component)
        return
    if args.command == "preprocess":
        run_preprocess(args.target)
        return
    run_benchmark(args.surrogate)


if __name__ == "__main__":
    main()
