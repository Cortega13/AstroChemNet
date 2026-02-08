"""Compress tracer `.npy` outputs into a single HDF5 with `/small` and `/large` groups."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PHYSICAL_KEYS = ("density", "temperature", "av", "rad_field")


@dataclass(frozen=True)
class TracerPayload:
    """In-memory representation of one tracer `.npy` payload."""

    tracer_id: int
    columns: list[str]
    data: np.ndarray


def _extract_tracer_id(path: Path) -> int:
    """Parse tracer id from a `tracer_<id>_<label>.npy` filename."""
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "tracer":
        raise ValueError(f"Unrecognized tracer filename: {path.name}")
    return int(parts[1])


def _load_payload(path: Path) -> TracerPayload:
    """Load a tracer `.npy` payload into memory."""
    payload = np.load(path, allow_pickle=True).item()
    columns = [str(c) for c in list(payload["columns"])]
    data = np.asarray(payload["data"], dtype=float)
    return TracerPayload(tracer_id=_extract_tracer_id(path), columns=columns, data=data)


def _canonicalize_columns(columns: list[str]) -> tuple[list[str], dict[str, int]]:
    """Return canonical column order and a name->source-index map."""
    col_to_idx = {name: i for i, name in enumerate(columns)}

    time_key = "time" if "time" in col_to_idx else "time_years"
    if time_key not in col_to_idx:
        raise ValueError(f"Missing time column in columns={columns[:10]} ...")

    for key in PHYSICAL_KEYS:
        if key not in col_to_idx:
            raise ValueError(f"Missing required physical column '{key}'")

    known = {time_key, *PHYSICAL_KEYS}
    species = [c for c in columns if c not in known]
    if not species:
        raise ValueError("No species columns detected (expected at least 1)")

    canonical = ["time", *PHYSICAL_KEYS, *species]
    index_map = dict(col_to_idx)

    index_map["time"] = col_to_idx[time_key]

    return canonical, index_map


def _reorder_data(
    payload: TracerPayload,
    target_columns: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    """Return reordered (columns, data) matching `target_columns` (or establish it)."""
    canonical, index_map = _canonicalize_columns(payload.columns)

    if target_columns is None:
        target_columns = canonical

    if len(canonical) != len(target_columns):
        raise ValueError(
            "Column mismatch across tracers. "
            f"expected_n={len(target_columns)} got_n={len(canonical)} "
            f"(file tracer_id={payload.tracer_id})"
        )

    try:
        src_indices = [index_map[c] for c in target_columns]
    except KeyError as exc:
        raise ValueError(
            "Column mismatch across tracers. "
            f"missing={exc.args[0]} (file tracer_id={payload.tracer_id})"
        ) from exc
    data = payload.data[:, src_indices]
    return target_columns, data


def _matched_tracer_ids(input_dir: Path) -> list[int]:
    """Return tracer ids that have both small and large outputs."""
    small_ids = {
        _extract_tracer_id(p) for p in sorted(input_dir.glob("tracer_*_small.npy"))
    }
    large_ids = {
        _extract_tracer_id(p) for p in sorted(input_dir.glob("tracer_*_large.npy"))
    }
    return sorted(small_ids & large_ids)


def _choose_chunk_rows(n_cols: int, target_chunk_bytes: int = 1_000_000) -> int:
    """Choose a chunk row count for ~1MB float32 chunks."""
    if n_cols <= 0:
        return 1
    rows = max(1, target_chunk_bytes // (n_cols * 4))
    return int(min(max(rows, 1), 4096))


def _append_rows(
    grp: h5py.Group,
    data_block: np.ndarray,
    tracer_id: int,
    tracer_ptr_rows: list[tuple[int, int, int]],
) -> None:
    """Append a tracer block to `data` and `tracer_id` datasets and record pointer."""
    dset = grp["data"]
    tid = grp["tracer_id"]

    start = int(dset.shape[0])
    n = int(data_block.shape[0])
    if n == 0:
        return

    dset.resize((start + n, dset.shape[1]))
    tid.resize((start + n,))

    dset[start : start + n] = data_block
    tid[start : start + n] = np.full((n,), tracer_id, dtype=np.int64)

    tracer_ptr_rows.append((int(tracer_id), start, n))


def _init_group(
    h5: h5py.File,
    group_name: str,
    columns: list[str],
) -> h5py.Group:
    """Create and return an output group with extendable compressed datasets."""
    if group_name in h5:
        del h5[group_name]
    grp = h5.create_group(group_name)

    str_dt = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset("columns", data=np.asarray(columns, dtype=object), dtype=str_dt)

    n_cols = len(columns)
    chunk_rows = _choose_chunk_rows(n_cols)

    compression = "gzip"
    compression_level = 4

    grp.create_dataset(
        "data",
        shape=(0, n_cols),
        maxshape=(None, n_cols),
        dtype=np.float32,
        chunks=(chunk_rows, n_cols),
        compression=compression,
        compression_opts=int(compression_level),
        shuffle=True,
    )
    grp.create_dataset(
        "tracer_id",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=(max(1024, chunk_rows),),
        compression=compression,
        compression_opts=int(compression_level),
        shuffle=True,
    )

    grp.attrs["compression"] = str(compression)
    grp.attrs["compression_level"] = int(compression_level)
    grp.attrs["n_columns"] = int(n_cols)
    grp.attrs["chunk_rows"] = int(chunk_rows)

    return grp


def compress_tracers_to_h5(
    input_dir: Path,
    output_path: Path,
) -> None:
    """Write `output_path` from all tracer pairs found in `input_dir`."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracer_ids = _matched_tracer_ids(input_dir)
    if not tracer_ids:
        raise FileNotFoundError(
            f"No matched tracer pairs found in {input_dir} (expected tracer_*_small.npy and tracer_*_large.npy)"
        )

    created_utc = datetime.now(timezone.utc).isoformat()

    first_id = tracer_ids[0]
    small_first = _load_payload(input_dir / f"tracer_{first_id}_small.npy")
    large_first = _load_payload(input_dir / f"tracer_{first_id}_large.npy")

    small_cols, small_data0 = _reorder_data(small_first, None)
    large_cols, large_data0 = _reorder_data(large_first, None)

    with h5py.File(output_path, "w") as h5:
        h5.attrs["created_utc"] = created_utc
        h5.attrs["source_dir"] = str(input_dir)
        h5.attrs["schema"] = "stacked_rows_with_tracer_ptr"

        grp_small = _init_group(h5, "small", small_cols)
        grp_large = _init_group(h5, "large", large_cols)

        small_target_cols = small_cols
        large_target_cols = large_cols

        ptr_small: list[tuple[int, int, int]] = []
        ptr_large: list[tuple[int, int, int]] = []

        _append_rows(
            grp_small, small_data0.astype(np.float32, copy=False), first_id, ptr_small
        )
        _append_rows(
            grp_large, large_data0.astype(np.float32, copy=False), first_id, ptr_large
        )

        for tracer_id in tqdm(tracer_ids[1:], desc=f"Compressing to {output_path}"):
            small = _load_payload(input_dir / f"tracer_{tracer_id}_small.npy")
            large = _load_payload(input_dir / f"tracer_{tracer_id}_large.npy")

            _, small_block = _reorder_data(small, small_target_cols)
            _, large_block = _reorder_data(large, large_target_cols)

            _append_rows(
                grp_small,
                small_block.astype(np.float32, copy=False),
                tracer_id,
                ptr_small,
            )
            _append_rows(
                grp_large,
                large_block.astype(np.float32, copy=False),
                tracer_id,
                ptr_large,
            )

        grp_small.create_dataset(
            "tracer_ptr", data=np.asarray(ptr_small, dtype=np.int64)
        )
        grp_large.create_dataset(
            "tracer_ptr", data=np.asarray(ptr_large, dtype=np.int64)
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(
        description="Compress tracer_*_{small,large}.npy outputs into one HDF5 file."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing tracer_*_small.npy and tracer_*_large.npy",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/tracers.h5"),
        help="Output HDF5 file path",
    )
    return p.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    compress_tracers_to_h5(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
