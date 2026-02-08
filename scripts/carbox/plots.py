"""Generate tracer plots from the compressed Carbox HDF5."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PLOT_SPECIES = [
    "H2",
    "CO",
    "C",
    "C+",
    "O",
    "H3+",
    "HCO+",
    "H3O+",
    "E-",
    "MG+",
    "H2O",
    "OH",
]

SMALL_KEY = "small"
LARGE_KEY = "large"


@dataclass
class TracerData:
    """Contain tracer plot data."""

    tracer_id: int
    time: np.ndarray
    physical: dict[str, np.ndarray]
    species: list[str]
    abundances: np.ndarray


class TracerPair(NamedTuple):
    """Hold small/large network outputs for one tracer id."""

    tracer_id: int
    small: TracerData
    large: TracerData


def _safe_log10(values: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Return log10(values) with a positive floor."""
    clipped = np.clip(values, float(floor), None)
    return np.log10(clipped)


def _decode_columns(raw: np.ndarray) -> list[str]:
    """Decode an HDF5 string dataset into python strings."""
    out: list[str] = []
    for item in raw.tolist():
        out.append(
            item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
        )
    return out


def _choose_time_key(columns: Sequence[str]) -> str:
    """Choose the best available time column key."""
    if "time" in columns:
        return "time"
    if "time_years" in columns:
        return "time_years"
    raise KeyError(f"Missing time column in columns={list(columns)[:10]}")


def _require_indices(columns: Sequence[str], required: Sequence[str]) -> list[int]:
    """Return indices for required column names."""
    col_to_idx = {name: i for i, name in enumerate(columns)}
    missing = [name for name in required if name not in col_to_idx]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return [int(col_to_idx[name]) for name in required]


def _require_dataset(grp: h5py.Group, name: str) -> h5py.Dataset:
    """Return a required dataset from an HDF5 group."""
    obj = grp.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise KeyError(f"Missing dataset '{name}'")
    return cast(h5py.Dataset, obj)


def _read_tracer_ptr(grp: h5py.Group) -> np.ndarray:
    """Read tracer_ptr as a 2D int64 array."""
    dset = _require_dataset(grp, "tracer_ptr")
    ptr = np.asarray(dset[()])
    if ptr.ndim != 2 or ptr.shape[1] != 3:
        raise ValueError(f"Unexpected tracer_ptr shape: {ptr.shape}")
    return ptr.astype(np.int64, copy=False)


def _tracer_ids_in_both(ptr_small: np.ndarray, ptr_large: np.ndarray) -> list[int]:
    """Return tracer ids present in both groups preserving small order."""
    small_ids = [int(v) for v in ptr_small[:, 0].tolist()]
    large_ids = {int(v) for v in ptr_large[:, 0].tolist()}
    out: list[int] = []
    for tid in small_ids:
        if tid in large_ids:
            out.append(tid)
    return out


def _slice_for_tracer(ptr: np.ndarray, tracer_id: int) -> tuple[int, int]:
    """Return (start, length) for a tracer id from tracer_ptr."""
    matches = np.where(ptr[:, 0] == int(tracer_id))[0]
    if matches.size == 0:
        raise KeyError(f"Tracer id {tracer_id} not found")
    row = ptr[int(matches[0])]
    return int(row[1]), int(row[2])


def parse_tracer_from_h5(h5_path: Path, key: str, tracer_id: int) -> TracerData:
    """Load a tracer from the compressed HDF5 into plot-ready arrays."""
    with h5py.File(h5_path, mode="r") as h5:
        obj = h5.get(key)
        if not isinstance(obj, h5py.Group):
            raise KeyError(f"Missing group '{key}' in {h5_path}")
        grp = cast(h5py.Group, obj)
        col_dset = _require_dataset(grp, "columns")
        columns = _decode_columns(np.asarray(col_dset[()]))
        ptr = _read_tracer_ptr(grp)
        start, length = _slice_for_tracer(ptr, tracer_id)
        data_dset = _require_dataset(grp, "data")
        data = np.asarray(data_dset[start : start + length], dtype=float)
    time_key = _choose_time_key(columns)
    phys_keys = ["density", "temperature", "av", "rad_field"]
    idx_time = int(columns.index(time_key))
    idx_phys = _require_indices(columns, phys_keys)
    time = np.asarray(data[:, idx_time], dtype=float)
    physical = {
        k: np.asarray(data[:, idx], dtype=float) for k, idx in zip(phys_keys, idx_phys)
    }
    species_start = 5
    species = [str(s) for s in columns[species_start:]]
    abundances = np.asarray(data[:, species_start:], dtype=float)
    return TracerData(
        tracer_id=int(tracer_id),
        time=time,
        physical=physical,
        species=species,
        abundances=abundances,
    )


def _plot_row_start(tracer: TracerData) -> int:
    """Return a safe plot start index."""
    if tracer.time.shape[0] > 1:
        return 1
    return 0


def plot_physical(ax: Axes, tracer: TracerData) -> None:
    """Plot physical parameters over time."""
    start = _plot_row_start(tracer)
    for key, label in (
        ("density", "Density"),
        ("temperature", "Temperature"),
        ("av", "Av"),
        ("rad_field", "Rad Field"),
    ):
        ax.plot(
            tracer.time[start:], _safe_log10(tracer.physical[key][start:]), label=label
        )
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("log10(Value)")
    ax.legend()


def plot_abundances(
    ax: Axes,
    tracer: TracerData,
    species_names: Sequence[str],
    colors: dict[str, object],
) -> None:
    """Plot selected species abundances over time."""
    start = _plot_row_start(tracer)
    index_by_name = {name: idx for idx, name in enumerate(tracer.species)}
    for name in species_names:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        ax.plot(
            tracer.time[start:],
            _safe_log10(tracer.abundances[start:, idx]),
            label=name,
            color=colors.get(name),
        )
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("log10(Abundance)")
    ax.legend(ncol=2, fontsize=8, loc="lower right", framealpha=0.85)


def build_color_map(species: Sequence[str]) -> dict[str, object]:
    """Assign stable colors to species names."""
    cmap = plt.get_cmap("tab20")
    colors: dict[str, object] = {}
    for i, name in enumerate(species):
        colors[name] = cmap(i % cmap.N)
    return colors


def render_tracer_plot(
    tracer_pair: TracerPair,
    species_names: Sequence[str],
    output_dir: Path,
    colors: dict[str, object],
) -> Path:
    """Create and save a tracer plot figure."""
    fig, axes = _create_tracer_figure()
    _plot_tracer_panels(axes, tracer_pair, species_names, colors)
    _set_tracer_panel_titles(axes, tracer_pair.tracer_id)
    return _save_tracer_figure(fig, output_dir, tracer_pair.tracer_id)


def _create_tracer_figure() -> tuple[Figure, Sequence[Axes]]:
    """Create a 3-panel tracer figure."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    return fig, axes


def _plot_tracer_panels(
    axes: Sequence[Axes],
    tracer_pair: TracerPair,
    species_names: Sequence[str],
    colors: dict[str, object],
) -> None:
    """Plot physical and abundance panels for a tracer."""
    plot_physical(axes[0], tracer_pair.small)
    plot_abundances(axes[1], tracer_pair.small, species_names, colors)
    plot_abundances(axes[2], tracer_pair.large, species_names, colors)


def _set_tracer_panel_titles(axes: Sequence[Axes], tracer_id: int) -> None:
    """Set titles for the three tracer panels."""
    axes[0].set_title(f"Tracer {tracer_id}")
    axes[1].set_title("Small network")
    axes[2].set_title("Large network")


def _save_tracer_figure(fig: Figure, output_dir: Path, tracer_id: int) -> Path:
    """Save a tracer figure to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tracer_{tracer_id}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _tracer_pairs_from_h5(h5_path: Path, n_tracers: int | None) -> list[TracerPair]:
    """Load tracer pairs from the compressed HDF5."""
    with h5py.File(h5_path, mode="r") as h5:
        small_obj = h5.get(SMALL_KEY)
        large_obj = h5.get(LARGE_KEY)
        if not isinstance(small_obj, h5py.Group):
            raise KeyError(f"Missing group '{SMALL_KEY}' in {h5_path}")
        if not isinstance(large_obj, h5py.Group):
            raise KeyError(f"Missing group '{LARGE_KEY}' in {h5_path}")
        ptr_small = _read_tracer_ptr(cast(h5py.Group, small_obj))
        ptr_large = _read_tracer_ptr(cast(h5py.Group, large_obj))
    ids = _tracer_ids_in_both(ptr_small, ptr_large)
    if n_tracers is not None:
        ids = ids[: int(max(0, n_tracers))]
    pairs: list[TracerPair] = []
    for tid in ids:
        small = parse_tracer_from_h5(h5_path, SMALL_KEY, tid)
        large = parse_tracer_from_h5(h5_path, LARGE_KEY, tid)
        pairs.append(TracerPair(tid, small, large))
    return pairs


def process_tracers_from_h5(
    h5_path: Path,
    output_dir: Path,
    n_tracers: int | None,
) -> None:
    """Generate plots from the compressed HDF5 for selected tracers."""
    tracer_pairs = _tracer_pairs_from_h5(h5_path, n_tracers)
    if not tracer_pairs:
        return
    species_names = PLOT_SPECIES
    colors = build_color_map(species_names)
    for pair in tracer_pairs:
        render_tracer_plot(pair, species_names, output_dir, colors)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot tracer physical parameters and abundances"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("data/carbox_grav.h5"),
        help="Compressed HDF5 file to read",
    )
    parser.add_argument(
        "--n-tracers",
        type=int,
        default=10,
        help="Max tracers to plot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/carbox/plots"),
        help="Directory to save plots",
    )
    return parser.parse_args()


def main() -> None:
    """Run tracer plotting."""
    args = parse_args()
    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(str(h5_path))
    process_tracers_from_h5(h5_path, Path(args.output_dir), int(args.n_tracers))


if __name__ == "__main__":
    main()
