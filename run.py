"""Runs train, preprocess, or benchmark from one entrypoint; example: `python run.py train`; assumptions: dataclass configs in `src/configs` define default aliases."""

import argparse

from src.benchmark import run_benchmark
from src.preprocess import run_preprocess
from src.train import run_training


def _build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser with train/preprocess/benchmark subcommands."""
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument("component", nargs="?", default="autoencoder_grav")

    prep_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    prep_parser.add_argument("source", nargs="?", default="grav")
    prep_parser.add_argument("method", nargs="?", default="initial")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarking")
    bench_parser.add_argument("surrogate", nargs="?", default="ae_emulator_grav")
    return parser


def main() -> None:
    """Dispatches the selected subcommand."""
    args = _build_parser().parse_args()
    if args.command == "train":
        run_training(args.component)
        return
    if args.command == "preprocess":
        run_preprocess(args.source, args.method)
        return
    run_benchmark(args.surrogate)


if __name__ == "__main__":
    main()
