"""Analyzes the Carbox gravitational collapse dataset and prepares train/val tensors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class CarboxPaths:
    """Defines standard paths for the Carbox dataset workflow."""

    h5_path: Path
    key: str
    dataset_name: str
    output_root: Path
    analysis_dir: Path
    preprocessed_dir: Path
    species_master_path: Path
    initial_abundances_path: Path


@dataclass(frozen=True)
class CarboxGroupInfo:
    """Stores metadata for the Carbox group layout."""

    n_rows: int
    n_cols: int
    tracer_ptr_shape: tuple[int, int]
    tracer_id_shape: tuple[int]
    columns: list[str]


@dataclass(frozen=True)
class ColumnStats:
    """Stores streaming summary statistics for a numeric column."""

    min_value: float
    max_value: float
    nan_count: int


@dataclass(frozen=True)
class DatasetSummary:
    """Stores dataset summary information for documentation."""

    h5_path: str
    key: str
    storage: str
    n_rows: int
    n_cols: int
    n_tracers: int
    tracer_length_stats: dict[str, float]
    renamed_columns: list[str]
    duplicate_renamed_columns: dict[str, int]
    phys_column_stats: dict[str, ColumnStats]
    time_column_stats: ColumnStats
    species_expected: int
    species_present: int
    abundance_minmax: list[float]
    abundance_out_of_bounds: dict[str, int]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for analysis and preprocessing."""
    parser = argparse.ArgumentParser(
        description="Analyze data/carbox_grav.h5 and build train/val tensors"
    )
    parser.add_argument("--h5-path", default="data/carbox_grav.h5")
    parser.add_argument("--key", default="large")
    parser.add_argument("--dataset-name", default="carbox_grav")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--train-split", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abundance-lower", type=float, default=1.0e-20)
    parser.add_argument("--abundance-upper", type=float, default=1.0)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--skip-species-file", action="store_true")
    parser.add_argument("--skip-stoichiometric", action="store_true")
    return parser.parse_args()


def _ensure_repo_on_syspath() -> None:
    """Ensure the repository root is importable as a module root."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _build_paths(args: argparse.Namespace) -> CarboxPaths:
    """Build standard paths for all outputs."""
    h5_path = Path(args.h5_path)
    dataset_name = str(args.dataset_name)
    output_root = Path(args.output_root)
    analysis_dir = output_root / "analysis" / dataset_name
    preprocessed_dir = output_root / "preprocessed" / dataset_name / "uclchem_grav"
    return CarboxPaths(
        h5_path=h5_path,
        key=str(args.key),
        dataset_name=dataset_name,
        output_root=output_root,
        analysis_dir=analysis_dir,
        preprocessed_dir=preprocessed_dir,
        species_master_path=Path("data/uclchem_species.txt"),
        initial_abundances_path=Path("data/initial_abundances.npy"),
    )


def _open_group(paths: CarboxPaths) -> h5py.Group:
    """Open and return the Carbox HDF5 group."""
    f = h5py.File(paths.h5_path, mode="r")
    obj = f.get(paths.key)
    if not isinstance(obj, h5py.Group):
        f.close()
        raise TypeError(f"Expected Group at '{paths.key}', got {type(obj)}")
    required = {"columns", "data", "tracer_id", "tracer_ptr"}
    if not required.issubset(set(obj.keys())):
        f.close()
        raise KeyError(
            f"Missing required datasets under '{paths.key}': {sorted(required)}"
        )
    return obj


def _decode_columns(raw: np.ndarray) -> list[str]:
    """Decode an HDF5 columns dataset into a list of strings."""
    out: list[str] = []
    for item in raw.tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _read_group_info(paths: CarboxPaths) -> CarboxGroupInfo:
    """Read group metadata without loading the full data matrix."""
    grp = _open_group(paths)
    data = grp["data"]
    columns_raw = grp["columns"][()]
    tracer_id = grp["tracer_id"]
    tracer_ptr = grp["tracer_ptr"]
    columns = _decode_columns(np.asarray(columns_raw))
    info = CarboxGroupInfo(
        n_rows=int(data.shape[0]),
        n_cols=int(data.shape[1]),
        tracer_ptr_shape=tuple(tracer_ptr.shape),
        tracer_id_shape=tuple(tracer_id.shape),
        columns=columns,
    )
    grp.file.close()
    return info


def _rename_mapping() -> dict[str, str]:
    """Return the canonical renaming mapping used for UCLCHEM column normalization."""
    return {
        "time": "Time",
        "density": "Density",
        "temperature": "gasTemp",
        "av": "Av",
        "rad_field": "Radfield",
        "H2COH+": "H3CO+",
        "H2COH": "H3CO",
        "H2CSH+": "H3CS+",
        "SISH+": "HSIS+",
        "HOSO+": "HSO2+",
        "OCSH+": "HOCS+",
        "HCOO": "HCO2",
        "HCOOH": "H2CO2",
        "CH2CO": "C2H2O",
        "CH2OH": "CH3O",
        "CH3CCH": "C3H4",
        "CH3CHO": "C2H4O",
        "CH3CN": "C2H3N",
        "CH3CNH": "C2H4N",
        "CH3OH": "CH4O",
        "CH3OH2+": "CH5O+",
        "CH3CNH+": "C2H4N+",
        "NH2CHO": "CH3NO",
        "HCO2H": "H2CO2",
        "HCNH": "H2CN",
        "NCCN": "N2C2",
        "Tracer": "Model",
        "radfield": "Radfield",
    }


def _rename_columns(columns: list[str]) -> list[str]:
    """Rename Carbox columns into canonical training names."""
    mapping = _rename_mapping()
    ordered = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    out: list[str] = []
    for col in columns:
        new = col.strip()
        for old, rep in ordered:
            if old in new:
                new = new.replace(old, rep)
        out.append(new)
    return out


def _count_duplicates(columns: list[str]) -> dict[str, int]:
    """Count duplicates in a column name list."""
    counts: dict[str, int] = {}
    for c in columns:
        counts[c] = counts.get(c, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def _read_species_master(paths: CarboxPaths) -> list[str]:
    """Read the canonical species list used by the project."""
    return paths.species_master_path.read_text().splitlines()


def _select_present_species(master: list[str], renamed_columns: list[str]) -> list[str]:
    """Select species present in the dataset based on the master list order."""
    present = set(renamed_columns)
    return [s for s in master if s in present]


def _write_text_lines(path: Path, lines: list[str]) -> None:
    """Write lines to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tracer_ptr_arrays(
    tracer_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract tracer_id, start, length arrays from tracer_ptr."""
    if tracer_ptr.ndim != 2 or tracer_ptr.shape[1] != 3:
        raise ValueError(f"Unexpected tracer_ptr shape: {tracer_ptr.shape}")
    ids = tracer_ptr[:, 0].astype(np.int64, copy=False)
    starts = tracer_ptr[:, 1].astype(np.int64, copy=False)
    lengths = tracer_ptr[:, 2].astype(np.int64, copy=False)
    return ids, starts, lengths


def _length_stats(lengths: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for tracer lengths."""
    values = lengths.astype(np.float64, copy=False)
    if values.size == 0:
        return {"min": 0.0, "p50": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def _build_name_to_indices(columns: list[str]) -> dict[str, list[int]]:
    """Build a mapping from name to one-or-more indices for duplicate handling."""
    out: dict[str, list[int]] = {}
    for i, c in enumerate(columns):
        out.setdefault(c, []).append(i)
    return out


def _nan_count(values: np.ndarray) -> int:
    """Count NaNs in a numeric array."""
    if not np.issubdtype(values.dtype, np.floating):
        return 0
    return int(np.isnan(values).sum())


def _finite_minmax(values: np.ndarray) -> tuple[float, float]:
    """Compute finite min and max values."""
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan"), float("nan")
    data = values[finite]
    return float(data.min()), float(data.max())


def _merge_stats(a: ColumnStats | None, b: ColumnStats) -> ColumnStats:
    """Merge two ColumnStats objects."""
    if a is None:
        return b
    mn = b.min_value if np.isnan(a.min_value) else min(a.min_value, b.min_value)
    mx = b.max_value if np.isnan(a.max_value) else max(a.max_value, b.max_value)
    if np.isnan(b.min_value):
        mn = a.min_value
    if np.isnan(b.max_value):
        mx = a.max_value
    return ColumnStats(
        min_value=float(mn), max_value=float(mx), nan_count=a.nan_count + b.nan_count
    )


def _column_stats_from_chunk(chunk: np.ndarray) -> ColumnStats:
    """Compute stats for a single chunk."""
    values = chunk.astype(np.float64, copy=False)
    mn, mx = _finite_minmax(values)
    return ColumnStats(min_value=mn, max_value=mx, nan_count=_nan_count(values))


def _iter_row_chunks(dset: h5py.Dataset, chunksize: int) -> Any:
    """Iterate over a 2D dataset in row chunks."""
    n = int(dset.shape[0])
    for start in range(0, n, int(chunksize)):
        stop = min(start + int(chunksize), n)
        yield dset[start:stop]


def _stream_stats_for_column(
    dset: h5py.Dataset, col_indices: list[int], chunksize: int
) -> ColumnStats:
    """Stream stats for a column that may be the sum of multiple indices."""
    acc: ColumnStats | None = None
    for chunk in _iter_row_chunks(dset, chunksize=chunksize):
        col = (
            chunk[:, col_indices].sum(axis=1)
            if len(col_indices) > 1
            else chunk[:, col_indices[0]]
        )
        acc = _merge_stats(acc, _column_stats_from_chunk(np.asarray(col)))
    if acc is None:
        return ColumnStats(min_value=float("nan"), max_value=float("nan"), nan_count=0)
    return acc


def _species_abundance_stream_stats(
    dset: h5py.Dataset,
    species_indices: np.ndarray,
    lower: float,
    upper: float,
    chunksize: int,
) -> tuple[list[float], dict[str, int]]:
    """Stream global abundance min/max and out-of-bounds counts."""
    min_val = float("nan")
    max_val = float("nan")
    below = 0
    above = 0
    for chunk in _iter_row_chunks(dset, chunksize=chunksize):
        abund = chunk[:, species_indices].astype(np.float64, copy=False)
        mn, mx = _finite_minmax(abund)
        if not np.isnan(mn):
            min_val = mn if np.isnan(min_val) else float(min(min_val, mn))
        if not np.isnan(mx):
            max_val = mx if np.isnan(max_val) else float(max(max_val, mx))
        below += int((abund < lower).sum())
        above += int((abund > upper).sum())
    return [float(min_val), float(max_val)], {
        "below_lower": below,
        "above_upper": above,
    }


def _analysis_summary(
    paths: CarboxPaths,
    info: CarboxGroupInfo,
    renamed: list[str],
    duplicates: dict[str, int],
    present_species: list[str],
    name_to_indices: dict[str, list[int]],
    chunksize: int,
    lower: float,
    upper: float,
) -> DatasetSummary:
    """Compute a dataset summary for documentation."""
    grp = _open_group(paths)
    dset = grp["data"]
    lengths = _load_tracer_lengths(grp)
    time_stats = _compute_time_stats(dset, name_to_indices, chunksize)
    phys_stats = _compute_phys_stats(dset, name_to_indices, chunksize)
    species_indices = _select_singleton_species_indices(
        name_to_indices, present_species
    )
    abund_minmax, bounds = _species_abundance_stream_stats(
        dset, species_indices, lower, upper, chunksize
    )
    summary = _build_dataset_summary(
        paths,
        info,
        renamed,
        duplicates,
        lengths,
        time_stats,
        phys_stats,
        present_species,
        abund_minmax,
        bounds,
    )
    grp.file.close()
    return summary


def _load_tracer_lengths(grp: h5py.Group) -> np.ndarray:
    """Load tracer length information from tracer_ptr."""
    tracer_ptr = np.asarray(grp["tracer_ptr"][()])
    _, _, lengths = _tracer_ptr_arrays(tracer_ptr)
    return lengths


def _require_indices(name_to_indices: dict[str, list[int]], name: str) -> list[int]:
    """Return indices for a required column name."""
    idxs = name_to_indices.get(name)
    if not idxs:
        raise KeyError(f"Missing required column after rename: {name}")
    return idxs


def _compute_time_stats(
    dset: h5py.Dataset, name_to_indices: dict[str, list[int]], chunksize: int
) -> ColumnStats:
    """Compute streaming stats for the time column."""
    return _stream_stats_for_column(
        dset, _require_indices(name_to_indices, "Time"), chunksize
    )


def _compute_phys_stats(
    dset: h5py.Dataset, name_to_indices: dict[str, list[int]], chunksize: int
) -> dict[str, ColumnStats]:
    """Compute streaming stats for physical parameter columns."""
    phys = ["Density", "Radfield", "Av", "gasTemp"]
    return {
        k: _stream_stats_for_column(
            dset, _require_indices(name_to_indices, k), chunksize
        )
        for k in phys
    }


def _select_singleton_species_indices(
    name_to_indices: dict[str, list[int]], species: list[str]
) -> np.ndarray:
    """Select species indices that map to exactly one column."""
    indices = [name_to_indices[s][0] for s in species if len(name_to_indices[s]) == 1]
    return np.asarray(indices, dtype=np.int64)


def _build_dataset_summary(
    paths: CarboxPaths,
    info: CarboxGroupInfo,
    renamed: list[str],
    duplicates: dict[str, int],
    lengths: np.ndarray,
    time_stats: ColumnStats,
    phys_stats: dict[str, ColumnStats],
    present_species: list[str],
    abund_minmax: list[float],
    bounds: dict[str, int],
) -> DatasetSummary:
    """Build a DatasetSummary from computed components."""
    return DatasetSummary(
        h5_path=str(paths.h5_path),
        key=paths.key,
        storage=f"h5py:{paths.key}{{columns,data,tracer_id,tracer_ptr}}",
        n_rows=info.n_rows,
        n_cols=info.n_cols,
        n_tracers=int(lengths.size),
        tracer_length_stats=_length_stats(lengths),
        renamed_columns=renamed,
        duplicate_renamed_columns=duplicates,
        phys_column_stats=phys_stats,
        time_column_stats=time_stats,
        species_expected=int(len(_read_species_master(paths))),
        species_present=int(len(present_species)),
        abundance_minmax=abund_minmax,
        abundance_out_of_bounds=bounds,
    )


def _render_summary_text(summary: DatasetSummary) -> str:
    """Render a human-readable summary to stdout."""
    lines: list[str] = []
    lines.append(f"HDF5: {summary.h5_path}")
    lines.append(f"Key: {summary.key}")
    lines.append(f"Storage: {summary.storage}")
    lines.append(f"Rows: {summary.n_rows}")
    lines.append(f"Cols: {summary.n_cols}")
    lines.append(f"Tracers: {summary.n_tracers}")
    lines.append(f"Tracer length stats: {summary.tracer_length_stats}")
    lines.append(f"Time stats: {summary.time_column_stats}")
    lines.append(f"Phys stats: {summary.phys_column_stats}")
    lines.append(
        f"Species present/expected: {summary.species_present}/{summary.species_expected}"
    )
    lines.append(f"Abundance minmax: {summary.abundance_minmax}")
    lines.append(f"Abundance bounds: {summary.abundance_out_of_bounds}")
    lines.append(f"Duplicate renamed columns: {len(summary.duplicate_renamed_columns)}")
    return "\n".join(lines)


def _render_summary_markdown(summary: DatasetSummary) -> str:
    """Render a markdown summary suitable for docs."""

    def _kv(name: str, value: Any) -> str:
        return f"- **{name}**: `{value}`"

    lines: list[str] = []
    lines.append(f"# Carbox dataset summary (`{summary.key}`)")
    lines.append("")
    lines.append(_kv("HDF5", summary.h5_path))
    lines.append(_kv("Storage", summary.storage))
    lines.append(_kv("Rows", summary.n_rows))
    lines.append(_kv("Columns", summary.n_cols))
    lines.append(_kv("Tracers", summary.n_tracers))
    lines.append("")
    lines.append("## Tracer lengths")
    for k, v in summary.tracer_length_stats.items():
        lines.append(_kv(k, v))
    lines.append("")
    lines.append("## Time")
    lines.append(_kv("min", summary.time_column_stats.min_value))
    lines.append(_kv("max", summary.time_column_stats.max_value))
    lines.append(_kv("nan_count", summary.time_column_stats.nan_count))
    lines.append("")
    lines.append("## Physical parameters")
    for name, stats in summary.phys_column_stats.items():
        lines.append(f"### {name}")
        lines.append(_kv("min", stats.min_value))
        lines.append(_kv("max", stats.max_value))
        lines.append(_kv("nan_count", stats.nan_count))
        lines.append("")
    lines.append("## Species coverage")
    lines.append(_kv("expected (project master)", summary.species_expected))
    lines.append(_kv("present in dataset", summary.species_present))
    lines.append("")
    lines.append("## Abundances")
    lines.append(_kv("min", summary.abundance_minmax[0]))
    lines.append(_kv("max", summary.abundance_minmax[1]))
    for k, v in summary.abundance_out_of_bounds.items():
        lines.append(_kv(k, v))
    lines.append("")
    lines.append("## Column renaming")
    lines.append(
        _kv("duplicate names after rename", len(summary.duplicate_renamed_columns))
    )
    if summary.duplicate_renamed_columns:
        lines.append("")
        lines.append("Duplicates:")
        for k, v in sorted(
            summary.duplicate_renamed_columns.items(), key=lambda kv: (-kv[1], kv[0])
        )[:50]:
            lines.append(_kv(k, v))
    lines.append("")
    return "\n".join(lines)


def _subset_indices_for_features(
    name_to_indices: dict[str, list[int]],
    phys: list[str],
    species: list[str],
) -> list[list[int]]:
    """Build a per-feature index list (supports duplicates)."""
    return [name_to_indices[name] for name in phys + species]


def _extract_features_from_slice(
    data_slice: np.ndarray, feature_indices: list[list[int]]
) -> np.ndarray:
    """Extract a feature matrix from a row slice with duplicate summation."""
    out = np.empty((data_slice.shape[0], len(feature_indices)), dtype=np.float32)
    for j, idxs in enumerate(feature_indices):
        if len(idxs) == 1:
            out[:, j] = data_slice[:, idxs[0]]
        else:
            out[:, j] = data_slice[:, idxs].sum(axis=1)
    return out


def _shift_physical_inplace(features: np.ndarray, num_phys: int) -> None:
    """Shift physical parameters by -1 in-place for a single tracer."""
    phys = features[:, :num_phys]
    phys[:-1] = phys[1:]
    if phys.shape[0] >= 2:
        phys[-1] = phys[-2]


def _clip_abundances_inplace(
    features: np.ndarray, num_phys: int, lower: float, upper: float
) -> None:
    """Clip abundances columns in-place."""
    abund = features[:, num_phys:]
    np.clip(abund, lower, upper, out=abund)


def _load_initial_abundances(paths: CarboxPaths) -> np.ndarray:
    """Load the project's initial abundances array."""
    return np.load(paths.initial_abundances_path)


def _map_initial_abundances(
    master_species: list[str],
    initial_abundances: np.ndarray,
    target_species: list[str],
) -> np.ndarray:
    """Map initial abundances from the master list onto a target species list."""
    if initial_abundances.ndim != 2 or initial_abundances.shape[1] != len(
        master_species
    ):
        raise ValueError(
            f"initial_abundances has shape {initial_abundances.shape}, expected (*, {len(master_species)})"
        )
    lookup = {name: i for i, name in enumerate(master_species)}
    row0 = initial_abundances[0].astype(np.float32, copy=False)
    out = np.zeros((len(target_species),), dtype=np.float32)
    for j, name in enumerate(target_species):
        if name in lookup:
            out[j] = row0[lookup[name]]
    return out


def _add_initial_row(
    features: np.ndarray, initial_abunds: np.ndarray, num_phys: int
) -> np.ndarray:
    """Prepend an initial-condition row to a tracer feature matrix."""
    out = np.empty((features.shape[0] + 1, features.shape[1]), dtype=np.float32)
    out[1:] = features
    out[0, :num_phys] = 0.0
    out[0, num_phys:] = initial_abunds
    return out


def _choose_fixed_length(lengths: np.ndarray) -> int:
    """Choose a fixed per-tracer length for tensorization."""
    if lengths.size == 0:
        raise ValueError("No tracers found")
    unique, counts = np.unique(lengths, return_counts=True)
    return int(unique[np.argmax(counts)])


def _select_tracer_rows(
    ids: np.ndarray, starts: np.ndarray, lengths: np.ndarray, fixed_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter tracer_ptr arrays to tracers with the fixed length."""
    mask = lengths == int(fixed_length)
    return ids[mask], starts[mask], lengths[mask]


def _split_tracers(
    n: int, train_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split tracer indices into train and validation subsets."""
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    split = int(float(train_split) * n)
    return indices[:split], indices[split:]


def _build_tensor_for_split(
    dset: h5py.Dataset,
    starts: np.ndarray,
    lengths: np.ndarray,
    feature_indices: list[list[int]],
    initial_row: np.ndarray,
    num_phys: int,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Build a 3D tensor for a set of tracers."""
    n_tracers = int(starts.size)
    t = int(lengths[0])
    f = int(len(feature_indices))
    tensor = torch.empty((n_tracers, t + 1, f), dtype=torch.float32)
    for i in range(n_tracers):
        start = int(starts[i])
        length = int(lengths[i])
        rows = dset[start : start + length]
        mat = _extract_features_from_slice(np.asarray(rows), feature_indices)
        mat = _add_initial_row(mat, initial_row, num_phys)
        _shift_physical_inplace(mat, num_phys)
        _clip_abundances_inplace(mat, num_phys, lower, upper)
        tensor[i].copy_(torch.from_numpy(mat))
    return tensor


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Save a tensor to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def _stoichiometric_elements() -> list[str]:
    """Return the chemical elements tracked for stoichiometry checks."""
    return ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]


def _element_patterns() -> dict[str, Any]:
    """Return compiled regex patterns for element matching."""
    import re

    return {
        "H": re.compile(r"H(?!E)(\d*)"),
        "HE": re.compile(r"HE(\d*)"),
        "C": re.compile(r"C(?!L)(\d*)"),
        "N": re.compile(r"N(\d*)"),
        "O": re.compile(r"O(\d*)"),
        "S": re.compile(r"S(?!I)(\d*)"),
        "SI": re.compile(r"SI(\d*)"),
        "MG": re.compile(r"MG(\d*)"),
        "CL": re.compile(r"CL(\d*)"),
    }


def _build_stoichiometric_matrix(species: list[str]) -> torch.Tensor:
    """Build a stoichiometric matrix with shape (num_species, num_elements)."""
    elems = _stoichiometric_elements()
    patterns = _element_patterns()
    mat = np.zeros((len(elems), len(species)), dtype=np.float32)
    cleaned = [s.replace("@", "").replace("#", "") for s in species]
    for e_idx, e in enumerate(elems):
        pat = patterns[e]
        for s_idx, name in enumerate(cleaned):
            match = pat.search(name)
            if match:
                mult = int(match.group(1)) if match.group(1) else 1
                mat[e_idx, s_idx] = float(mult)
    return torch.from_numpy(mat.T)


def _write_stoichiometric_matrix(path: Path, species: list[str]) -> None:
    """Write a stoichiometric matrix to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_build_stoichiometric_matrix(species), path)


def _physical_ranges_from_stats(
    stats: dict[str, ColumnStats],
) -> dict[str, list[float]]:
    """Convert ColumnStats into config-friendly range lists."""
    out: dict[str, list[float]] = {}
    for k, v in stats.items():
        out[k] = [float(v.min_value), float(v.max_value)]
    return out


def _config_yaml_text(
    dataset_name: str,
    raw_path: str,
    input_key: str,
    species_file: str,
    n_species: int,
    phys_ranges: dict[str, list[float]],
) -> str:
    """Render a dataset config YAML for the Carbox dataset."""
    density = phys_ranges.get("Density", [0.0, 1.0])
    radfield = phys_ranges.get("Radfield", [0.0, 1.0])
    av = phys_ranges.get("Av", [0.0, 1.0])
    gastemp = phys_ranges.get("gasTemp", [0.0, 1.0])
    return "\n".join(
        [
            f"name: {dataset_name}",
            f"raw_path: {raw_path}",
            f"input_key: {input_key}",
            "",
            "n_species: " + str(n_species),
            f"species_file: {species_file}",
            "initial_abundances: data/initial_abundances.npy",
            "",
            "phys: [Density, Radfield, Av, gasTemp]",
            "",
            "physical_parameters:",
            "  n_params: 4",
            "  ranges:",
            f"    Density: [{density[0]}, {density[1]}]",
            f"    Radfield: [{radfield[0]}, {radfield[1]}]",
            f"    Av: [{av[0]}, {av[1]}]",
            f"    gasTemp: [{gastemp[0]}, {gastemp[1]}]",
            "",
            "metadata_columns: [Index, Model, Time]",
            "columns_to_drop: []",
            "num_metadata: 3",
            "num_phys: 4",
            "",
            "abundances_clipping:",
            "  lower: 1.0e-20",
            "  upper: 1.0",
            "",
            f"stoichiometric_matrix_path: outputs/preprocessed/{dataset_name}/uclchem_grav/stoichiometric_matrix.pt",
            "",
            "train_split: 0.75",
            "seed: 42",
            "",
            "num_species: " + str(n_species),
            "",
        ]
    )


def _write_dataset_config(
    paths: CarboxPaths, n_species: int, phys_ranges: dict[str, list[float]]
) -> Path:
    """Write a dataset config YAML under configs/data."""
    cfg_path = Path("configs/data") / f"{paths.dataset_name}.yaml"
    text = _config_yaml_text(
        dataset_name=paths.dataset_name,
        raw_path=str(paths.h5_path),
        input_key=paths.key,
        species_file="data/carbox_species.txt",
        n_species=n_species,
        phys_ranges=phys_ranges,
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(text)
    return cfg_path


def _prepare_tensors(
    paths: CarboxPaths,
    renamed_columns: list[str],
    species: list[str],
    train_split: float,
    seed: int,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Prepare and save train/val tensors from the Carbox group layout."""
    grp = _open_group(paths)
    data = grp["data"]
    starts, lengths, fixed = _fixed_length_tracers(grp)
    train_idx, val_idx = _split_tracers(int(starts.size), train_split, seed)
    feature_indices, init_row, num_phys = _tensorization_inputs(
        paths, renamed_columns, species
    )
    _save_split_tensor(
        paths.preprocessed_dir / "uclchem_grav_train_preprocessed.pt",
        data,
        starts[train_idx],
        lengths[train_idx],
        feature_indices,
        init_row,
        num_phys,
        lower,
        upper,
    )
    _save_split_tensor(
        paths.preprocessed_dir / "uclchem_grav_val_preprocessed.pt",
        data,
        starts[val_idx],
        lengths[val_idx],
        feature_indices,
        init_row,
        num_phys,
        lower,
        upper,
    )
    grp.file.close()
    return _prep_metadata(
        fixed, starts.size, train_idx.size, val_idx.size, feature_indices, species
    )


def _fixed_length_tracers(grp: h5py.Group) -> tuple[np.ndarray, np.ndarray, int]:
    """Select tracers of the modal length."""
    tracer_ptr = np.asarray(grp["tracer_ptr"][()])
    ids, starts, lengths = _tracer_ptr_arrays(tracer_ptr)
    fixed = _choose_fixed_length(lengths)
    _, starts, lengths = _select_tracer_rows(ids, starts, lengths, fixed)
    return starts, lengths, fixed


def _tensorization_inputs(
    paths: CarboxPaths, renamed_columns: list[str], species: list[str]
) -> tuple[list[list[int]], np.ndarray, int]:
    """Build feature indices and initial row for tensorization."""
    name_to_indices = _build_name_to_indices(renamed_columns)
    phys = ["Density", "Radfield", "Av", "gasTemp"]
    feature_indices = _subset_indices_for_features(name_to_indices, phys, species)
    master = _read_species_master(paths)
    init = _load_initial_abundances(paths)
    init_row = _map_initial_abundances(master, init, species)
    return feature_indices, init_row, len(phys)


def _save_split_tensor(
    path: Path,
    dset: h5py.Dataset,
    starts: np.ndarray,
    lengths: np.ndarray,
    feature_indices: list[list[int]],
    init_row: np.ndarray,
    num_phys: int,
    lower: float,
    upper: float,
) -> None:
    """Build and save a single split tensor."""
    tensor = _build_tensor_for_split(
        dset, starts, lengths, feature_indices, init_row, num_phys, lower, upper
    )
    _save_tensor(path, tensor)


def _prep_metadata(
    fixed: int,
    n_tracers: int,
    n_train: int,
    n_val: int,
    feature_indices: list[list[int]],
    species: list[str],
) -> dict[str, Any]:
    """Build metadata for the preparation step."""
    return {
        "fixed_length": int(fixed),
        "n_tracers_used": int(n_tracers),
        "n_train_tracers": int(n_train),
        "n_val_tracers": int(n_val),
        "n_features": int(len(feature_indices)),
        "n_species": int(len(species)),
    }


def _main() -> None:
    """Run the end-to-end analysis and preparation workflow."""
    args = _parse_args()
    paths = _build_paths(args)
    if not paths.h5_path.exists():
        raise FileNotFoundError(str(paths.h5_path))

    info = _read_group_info(paths)
    renamed = _rename_columns(info.columns)
    duplicates = _count_duplicates(renamed)
    name_to_indices = _build_name_to_indices(renamed)

    master = _read_species_master(paths)
    present_species = _select_present_species(master, renamed)

    summary = _analysis_summary(
        paths,
        info,
        renamed,
        duplicates,
        present_species,
        name_to_indices,
        chunksize=int(args.chunksize),
        lower=float(args.abundance_lower),
        upper=float(args.abundance_upper),
    )

    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths.analysis_dir / "summary.json", asdict(summary))
    (paths.analysis_dir / "summary.txt").write_text(
        _render_summary_text(summary) + "\n"
    )
    (paths.analysis_dir / "summary.md").write_text(
        _render_summary_markdown(summary) + "\n"
    )
    print(_render_summary_text(summary))

    if not bool(args.skip_species_file):
        species_path = Path("data/carbox_species.txt")
        _write_text_lines(species_path, present_species)

    if not bool(args.skip_config):
        phys_ranges = _physical_ranges_from_stats(summary.phys_column_stats)
        _write_dataset_config(
            paths, n_species=len(present_species), phys_ranges=phys_ranges
        )

    if not bool(args.skip_stoichiometric):
        _write_stoichiometric_matrix(
            paths.preprocessed_dir / "stoichiometric_matrix.pt", present_species
        )

    if not bool(args.skip_prepare):
        prep = _prepare_tensors(
            paths,
            renamed,
            present_species,
            train_split=float(args.train_split),
            seed=int(args.seed),
            lower=float(args.abundance_lower),
            upper=float(args.abundance_upper),
        )
        _write_json(paths.analysis_dir / "prepare.json", prep)


if __name__ == "__main__":
    _main()
