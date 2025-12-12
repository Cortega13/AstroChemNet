"""Compare training runs."""

import json
from pathlib import Path

import pandas as pd


def compare_runs(component_names: list[str]):
    """Compare multiple training runs."""
    root = Path(__file__).parent.parent
    rows = []

    for name in component_names:
        summary_path = root / f"outputs/weights/{name}/summary.json"
        if not summary_path.exists():
            print(f"Warning: {name} not found")
            continue

        with open(summary_path) as f:
            summary = json.load(f)
        summary["component"] = name
        rows.append(summary)

    df = pd.DataFrame(rows)
    print(df.sort_values("best_val_loss"))
    return df


def plot_learning_curves(component_names: list[str]):
    """Plot learning curves for multiple components."""
    import matplotlib.pyplot as plt

    root = Path(__file__).parent.parent
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name in component_names:
        metrics_path = root / f"outputs/weights/{name}/metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        epochs = [m["epoch"] for m in metrics]
        train_loss = [m["train_loss"] for m in metrics]
        val_loss = [m["val_loss"] for m in metrics]

        ax1.plot(epochs, train_loss, label=name)
        ax2.plot(epochs, val_loss, label=name)

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(root / "outputs/comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys

    compare_runs(sys.argv[1:])
