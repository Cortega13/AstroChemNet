"""Preprocessing script for Carbox gravitational collapse dataset."""

import json
import os

import numpy as np

from .uclchem_grav_preprocessing import (
    get_output_dir,
    round_down_sigfigs,
    round_up_sigfigs,
    save_metadata,
    save_species,
    save_stoichiometric_matrix,
    save_train_val_split,
)

SEED = 42


def preprocess_carbox_grav(
    dataset_name: str = "carbox_grav", force: bool = False
) -> None:
    """Preprocess Carbox gravitational collapse dataset.

    Input files:
      - data/carbox_grav.npy
      - data/carbox_grav_columns.json

    Output (outputs/preprocessed/<dataset_name>/):
      - train.npy / val.npy with canonical columns:
        Index, Model, Time, Density, Radfield, Av, gasTemp, <species...>
      - columns.json (mapping for the canonical arrays)
      - species.json, stoichiometric_matrix.npy
      - physical_parameter_ranges.json, train_val_split.json, metadata.json
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_dir = get_output_dir(project_root, dataset_name)
    if os.path.exists(output_dir) and not force:
        print(
            f"Preprocessing output already exists at {output_dir}\nUse --force to overwrite"
        )
        return
    os.makedirs(output_dir, exist_ok=True)

    source_file = "data/carbox_grav.npy"
    raw = np.load(os.path.join(project_root, source_file), mmap_mode="r")
    with open(os.path.join(project_root, "data/carbox_grav_columns.json"), "r") as f:
        mapping = json.load(f)
    columns = [mapping[str(i)] for i in range(raw.shape[1])]
    species = columns[6:]

    model_ids = raw[:, 0].astype(np.int32)
    all_models = np.unique(model_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_models)
    split_ratio = 0.75
    split_idx = int(len(all_models) * split_ratio)
    train_models, val_models = all_models[:split_idx], all_models[split_idx:]
    is_train = np.zeros(int(model_ids.max()) + 1, dtype=bool)
    is_train[train_models] = True
    train_mask = is_train[model_ids]
    val_mask = ~train_mask

    def _build(mask: np.ndarray) -> np.ndarray:
        x = raw[mask]
        out = np.empty((len(x), 7 + len(species)), dtype=np.float32)
        out[:, 0] = np.arange(len(x), dtype=np.float32)
        out[:, 1] = x[:, 0]  # Model (tracer_id)
        out[:, 2] = x[:, 1]  # Time (time_years)
        out[:, 3] = x[:, 2]  # Density
        out[:, 4] = x[:, 5]  # Radfield
        out[:, 5] = x[:, 4]  # Av
        out[:, 6] = x[:, 3]  # gasTemp (temperature)
        out[:, 7:] = x[:, 6:]

        # Align physical parameters with next-timestep targets (match UCLCHEM convention)
        model = out[:, 1].astype(np.int32)
        same_next = model[:-1] == model[1:]
        out[:-1, 3:7][same_next] = out[1:, 3:7][same_next]
        ends = np.flatnonzero(~same_next)
        ends = ends[ends > 0]
        out[ends, 3:7] = out[ends - 1, 3:7]
        out[-1, 3:7] = out[-2, 3:7]
        return out

    train_data = _build(train_mask)
    val_data = _build(val_mask)

    phys_cols = ["Density", "Radfield", "Av", "gasTemp"]
    units = {
        "Density": "H nuclei per cm^3",
        "Radfield": "Habing field",
        "Av": "Magnitudes",
        "gasTemp": "Kelvin",
    }
    phys_ranges = {}
    for p, idx in zip(phys_cols, [3, 4, 5, 6]):
        mn, mx = float(train_data[:, idx].min()), float(train_data[:, idx].max())
        phys_ranges[p] = {
            "min": round_down_sigfigs(mn),
            "max": round_up_sigfigs(mx),
            "unit": units[p],
        }
    with open(os.path.join(output_dir, "physical_parameter_ranges.json"), "w") as f:
        json.dump(phys_ranges, f, indent=2)

    out_columns = ["Index", "Model", "Time"] + phys_cols + species
    with open(os.path.join(output_dir, "columns.json"), "w") as f:
        json.dump({str(i): c for i, c in enumerate(out_columns)}, f, indent=2)

    np.save(os.path.join(output_dir, "train.npy"), train_data)
    np.save(os.path.join(output_dir, "val.npy"), val_data)
    save_species(output_dir, species)
    save_stoichiometric_matrix(output_dir, species)
    save_train_val_split(
        output_dir,
        [int(m) for m in train_models],
        [int(m) for m in val_models],
        seed=SEED,
    )
    save_metadata(
        output_dir,
        dataset_name,
        source_file,
        num_species=len(species),
        num_physical_params=len(phys_cols),
        num_train_samples=len(train_data),
        num_val_samples=len(val_data),
        train_val_split_ratio=split_ratio,
    )
