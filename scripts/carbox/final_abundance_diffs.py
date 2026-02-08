"""Summarize final-time abundance Δlog10 = log10(large) - log10(small) across all tracers.

Example usage: python scripts/carbox/final_abundance_diffs.py --h5-path data/carbox_gravitational_collapse.h5 --top-k 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

SMALL_KEY = "small"
LARGE_KEY = "large"
SPECIES_START = 5


def safe_log10(values: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Compute log10(values) with a floor to avoid -inf."""
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


def _ptr_map(ptr: np.ndarray) -> dict[int, tuple[int, int]]:
    """Build a tracer_id -> (start, length) lookup."""
    out: dict[int, tuple[int, int]] = {}
    for tracer_id, start, length in ptr.tolist():
        out[int(tracer_id)] = (int(start), int(length))
    return out


def _common_species(
    small_columns: list[str],
    large_columns: list[str],
    species_start: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (common_species, small_indices, large_indices) aligned by species."""
    small_species = [str(s) for s in small_columns[species_start:]]
    large_species = [str(s) for s in large_columns[species_start:]]

    small_idx = {name: i for i, name in enumerate(small_species)}
    large_idx = {name: i for i, name in enumerate(large_species)}

    common: list[str] = []
    small_indices: list[int] = []
    large_indices: list[int] = []
    for name in small_species:
        j = large_idx.get(name)
        if j is None:
            continue
        common.append(name)
        small_indices.append(int(small_idx[name]))
        large_indices.append(int(j))

    return (
        common,
        np.asarray(small_indices, dtype=np.int64),
        np.asarray(large_indices, dtype=np.int64),
    )


def _last_row_indices(
    ptr_by_id: dict[int, tuple[int, int]],
    tracer_ids: list[int],
) -> np.ndarray:
    """Return last-row indices in `data` for each tracer id."""
    out = np.empty((len(tracer_ids),), dtype=np.int64)
    for i, tracer_id in enumerate(tracer_ids):
        start, length = ptr_by_id[int(tracer_id)]
        out[i] = int(start + length - 1)
    return out


def _read_group_columns(grp: h5py.Group) -> list[str]:
    """Read and decode the `columns` dataset."""
    dset = _require_dataset(grp, "columns")
    raw = np.asarray(dset[()])
    return _decode_columns(raw)


def _open_required_groups(
    h5: h5py.File,
    *,
    small_key: str,
    large_key: str,
) -> tuple[h5py.Group, h5py.Group]:
    """Return required small/large groups from an HDF5 file."""
    small_obj = h5.get(small_key)
    large_obj = h5.get(large_key)
    if not isinstance(small_obj, h5py.Group):
        raise KeyError(f"Missing group '{small_key}'")
    if not isinstance(large_obj, h5py.Group):
        raise KeyError(f"Missing group '{large_key}'")
    return cast(h5py.Group, small_obj), cast(h5py.Group, large_obj)


def _limit_tracer_ids(tracer_ids: list[int], max_tracers: int | None) -> list[int]:
    """Return tracer ids optionally truncated to max_tracers."""
    if max_tracers is None:
        return tracer_ids
    return tracer_ids[: int(max(0, max_tracers))]


def _tracer_ids_present_in_both_groups(
    grp_small: h5py.Group,
    grp_large: h5py.Group,
    *,
    max_tracers: int | None,
) -> list[int]:
    """Return tracer ids in both groups with optional truncation."""
    ptr_small = _read_tracer_ptr(grp_small)
    ptr_large = _read_tracer_ptr(grp_large)
    tracer_ids = _tracer_ids_in_both(ptr_small, ptr_large)
    return _limit_tracer_ids(tracer_ids, max_tracers)


def _common_species_indices(
    grp_small: h5py.Group,
    grp_large: h5py.Group,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return aligned common species and indices into small/large species blocks."""
    cols_small = _read_group_columns(grp_small)
    cols_large = _read_group_columns(grp_large)
    return _common_species(cols_small, cols_large, SPECIES_START)


def _collect_final_deltas(
    h5_path: Path,
    *,
    small_key: str,
    large_key: str,
    floor: float,
    max_tracers: int | None,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Collect Δlog10(large-small) for final abundances aligned by species."""
    with h5py.File(h5_path, mode="r") as h5:
        grp_small, grp_large = _open_required_groups(
            h5,
            small_key=small_key,
            large_key=large_key,
        )
        tracer_ids = _tracer_ids_present_in_both_groups(
            grp_small,
            grp_large,
            max_tracers=max_tracers,
        )
        if not tracer_ids:
            return [], np.empty((0, 0), dtype=np.float32), []

        common, idx_small, idx_large = _common_species_indices(grp_small, grp_large)
        if not common:
            return [], np.empty((0, 0), dtype=np.float32), []

        deltas = _compute_delta_matrix(
            grp_small,
            grp_large,
            tracer_ids,
            idx_small,
            idx_large,
            floor=float(floor),
        )
        return common, deltas, tracer_ids


def _compute_delta_matrix(
    grp_small: h5py.Group,
    grp_large: h5py.Group,
    tracer_ids: list[int],
    idx_small: np.ndarray,
    idx_large: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """Compute Δlog10 matrix with shape (n_tracers, n_species)."""
    data_small = _require_dataset(grp_small, "data")
    data_large = _require_dataset(grp_large, "data")
    last_small = _last_row_indices(_ptr_map(_read_tracer_ptr(grp_small)), tracer_ids)
    last_large = _last_row_indices(_ptr_map(_read_tracer_ptr(grp_large)), tracer_ids)

    small_rows = np.asarray(data_small[last_small, SPECIES_START:], dtype=float)
    large_rows = np.asarray(data_large[last_large, SPECIES_START:], dtype=float)

    small_vals = np.asarray(small_rows[:, idx_small], dtype=float)
    large_vals = np.asarray(large_rows[:, idx_large], dtype=float)
    delta = safe_log10(large_vals, floor=floor) - safe_log10(small_vals, floor=floor)
    return np.asarray(delta, dtype=np.float32)


@dataclass(frozen=True)
class SpeciesStats:
    """Hold aggregated delta statistics for one species."""

    name: str
    n_tracers: int
    mean_delta: float
    std_delta: float
    median_delta: float
    p05_delta: float
    p95_delta: float
    mean_abs_delta: float
    median_abs_delta: float
    max_abs_delta: float
    frac_over_threshold: float


def _summarize_species(
    name: str,
    deltas: np.ndarray,
    *,
    threshold_oom: float,
) -> SpeciesStats:
    """Compute aggregated statistics for one species."""
    n_tracers = int(deltas.shape[0])
    abs_delta = np.abs(deltas)

    qs = np.quantile(deltas, [0.05, 0.5, 0.95])
    frac = float(np.mean(abs_delta >= float(threshold_oom)))
    return SpeciesStats(
        name=str(name),
        n_tracers=n_tracers,
        mean_delta=float(np.mean(deltas)),
        std_delta=float(np.std(deltas)),
        median_delta=float(qs[1]),
        p05_delta=float(qs[0]),
        p95_delta=float(qs[2]),
        mean_abs_delta=float(np.mean(abs_delta)),
        median_abs_delta=float(np.median(abs_delta)),
        max_abs_delta=float(np.max(abs_delta)),
        frac_over_threshold=frac,
    )


def _summarize_all_species(
    species: list[str],
    delta_matrix: np.ndarray,
    *,
    threshold_oom: float,
) -> list[SpeciesStats]:
    """Compute per-species stats across all tracers."""
    stats: list[SpeciesStats] = []
    for i, name in enumerate(species):
        stats.append(
            _summarize_species(
                str(name),
                np.asarray(delta_matrix[:, i], dtype=float),
                threshold_oom=float(threshold_oom),
            )
        )
    return stats


def _print_summary(
    stats: list[SpeciesStats],
    *,
    n_tracers: int,
    threshold_oom: float,
    top_k: int | None,
) -> None:
    """Print aggregated stats sorted by max |Δlog10|."""
    print(f"Tracers processed: {n_tracers}")
    print(f"Common species: {len(stats)}")
    print(f"Threshold |Δlog10| for fraction: {threshold_oom}")

    ordered = sorted(stats, key=lambda s: s.max_abs_delta, reverse=True)
    if top_k is not None:
        ordered = ordered[: int(top_k)]

    header = "species        max|Δ|  median|Δ|  mean|Δ|   meanΔ   p05Δ   p95Δ   frac(|Δ|>=thr)"
    print(header)
    for s in ordered:
        print(
            f"{s.name:<12}  {s.max_abs_delta:7.2f}  {s.median_abs_delta:9.2f}  {s.mean_abs_delta:7.2f}  "
            f"{s.mean_delta:+6.2f}  {s.p05_delta:+6.2f}  {s.p95_delta:+6.2f}  {s.frac_over_threshold:16.3f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize final-time abundance differences between small and large networks "
            "in orders-of-magnitude (Δlog10) aggregated across all tracers."
        )
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("data/carbox_gravitational_collapse.h5"),
        help="Compressed HDF5 file produced by compress_tracer_outputs_to_h5.py",
    )
    parser.add_argument(
        "--max-tracers",
        type=int,
        default=0,
        help="Max tracers to process (0 = all)",
    )
    parser.add_argument(
        "--threshold-oom",
        type=float,
        default=1e-4,
        help="Minimum |Δlog10| to print (1.0 = 1 order-of-magnitude)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Max species rows to print (sorted by max |Δlog10|); use 0 for unlimited",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=1e-20,
        help="Floor for abundances before taking log10",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(str(h5_path))

    top_k = None if int(args.top_k) == 0 else int(args.top_k)
    max_tracers = None if int(args.max_tracers) == 0 else int(args.max_tracers)

    species, deltas, tracer_ids = _collect_final_deltas(
        h5_path,
        small_key=SMALL_KEY,
        large_key=LARGE_KEY,
        floor=float(args.floor),
        max_tracers=max_tracers,
    )
    if deltas.size == 0:
        return

    stats = _summarize_all_species(
        species,
        deltas,
        threshold_oom=float(args.threshold_oom),
    )
    _print_summary(
        stats,
        n_tracers=int(len(tracer_ids)),
        threshold_oom=float(args.threshold_oom),
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
