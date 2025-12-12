"""Initial preprocessor for raw UCLCHEM data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Chemical elements tracked for stoichiometric matrix
CHEMICAL_ELEMENTS: Final[list[str]] = [
    "H",
    "HE",
    "C",
    "N",
    "O",
    "S",
    "SI",
    "MG",
    "CL",
]

# Regex patterns for element matching in species names
ELEMENT_PATTERNS: Final[dict[str, re.Pattern]] = {
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


def rename_columns(columns):
    """Renames column names containing chemical species using substring replacement."""
    name_mapping = {
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

    sorted_mapping = sorted(name_mapping.items(), key=lambda x: -len(x[0]))

    columns = [col.strip() for col in columns]
    new_columns = []
    for col in columns:
        new_col = col
        for old, new in sorted_mapping:
            if old in new_col:
                new_col = new_col.replace(old, new)
        new_columns.append(new_col)

    return new_columns


class InitialPreprocessor:
    """Preprocesses raw UCLCHEM data into cleaned PyTorch tensors."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the preprocessor."""
        self.cfg = cfg

    def load_initial_abundances(self, species: list[str]) -> pd.DataFrame:
        """Loads initial abundances with uninitialized information."""
        initial_abundances = np.load(self.cfg.initial_abundances)
        df_init = pd.DataFrame(initial_abundances, columns=species)
        df_init["Radfield"] = 0
        df_init["Time"] = 0
        df_init["Av"] = 0
        df_init["gasTemp"] = 0
        df_init["Density"] = 0
        return df_init

    def _load_species(self) -> list[str]:
        """Loads the list of chemical species from file."""
        return np.loadtxt(
            self.cfg.species_file, dtype=str, delimiter=" ", comments=None
        ).tolist()

    def _load_raw_data(self) -> pd.DataFrame:
        """Loads raw data from HDF5 file."""
        input_path = Path(self.cfg.raw_path)
        # Default key to 'data' if not specified, as it was previously in preprocessing.input_key
        key = getattr(self.cfg, "input_key", "data")
        df = pd.read_hdf(input_path, key=key)
        if isinstance(df, pd.DataFrame):
            return df
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames columns and drops unwanted ones."""
        df.columns = rename_columns(df.columns)
        if self.cfg.columns_to_drop:
            df = df.drop(columns=self.cfg.columns_to_drop, errors="ignore")
        if hasattr(self.cfg, "min_radfield"):
            df["Radfield"] = np.maximum(df["Radfield"], self.cfg.min_radfield)
        return df

    def _process_trajectories(
        self, df: pd.DataFrame, species: list[str]
    ) -> pd.DataFrame:
        """Adds initial conditions and shifts physical parameters for each trajectory."""
        df_init = self.load_initial_abundances(species)
        output_chunks = []
        df.sort_values(by=["Model", "Time"], inplace=True)

        for _, tdf in df.groupby("Model", sort=False):
            tdf = tdf.reset_index(drop=True)
            df_init["Model"] = tdf.iloc[0]["Model"]
            tdf = pd.concat([df_init, tdf], ignore_index=True)

            physical = tdf[self.cfg.phys].shift(-1)
            physical.iloc[-1] = physical.iloc[-2]
            tdf.loc[:, self.cfg.phys] = physical.values
            output_chunks.append(tdf)

        return pd.concat(output_chunks, ignore_index=True)

    def _clip_abundances(self, df: pd.DataFrame, species: list[str]) -> pd.DataFrame:
        """Clips species abundances to configured range."""
        df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
        df[species] = df[species].clip(
            lower=self.cfg.abundances_clipping.lower,
            upper=self.cfg.abundances_clipping.upper,
        )
        return df

    def _get_split_tracers(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Splits tracers into training and validation sets."""
        np.random.seed(self.cfg.seed)
        tracers = df["Model"].unique()
        np.random.shuffle(tracers)
        split_idx = int(len(tracers) * self.cfg.train_split)
        return tracers[:split_idx], tracers[split_idx:]

    def _to_3d_tensor(
        self, df: pd.DataFrame, tracers: np.ndarray, species: list[str]
    ) -> torch.Tensor:
        """Converts dataframe to 3D tensor (Tracer, Time, Features)."""
        df = df.sort_values(by=["Model", "Time"])
        grouped = df.groupby("Model")
        tensors = []

        for tracer in tracers:
            if tracer in grouped.groups:
                group = grouped.get_group(tracer)
                data = group[self.cfg.phys + species].values
                tensors.append(data)

        return torch.tensor(np.stack(tensors), dtype=torch.float32)

    def _save_data(
        self,
        output_dir: Path,
        train: torch.Tensor,
        val: torch.Tensor,
    ) -> None:
        """Saves processed tensors to .pt files."""
        train_filename = getattr(
            self.cfg.output, "train_tensor", "initial_train_preprocessed.pt"
        )
        val_filename = getattr(
            self.cfg.output, "val_tensor", "initial_val_preprocessed.pt"
        )

        output_path_train = output_dir / train_filename
        output_path_val = output_dir / val_filename

        torch.save(train, output_path_train)
        torch.save(val, output_path_val)

        print(f"Saved train tensor to: {output_path_train}")
        print(f"Saved val tensor to: {output_path_val}")

    def _build_stoichiometric_matrix(
        self, species: list[str], output_dir: Path
    ) -> np.ndarray:
        """Build stoichiometric matrix S where x @ S yields elemental abundances.

        Returns (num_species, num_elements) matrix.

        Note: Excludes electrons (not conserved by UCLCHEM) and SURFACE/BULK species.
        """
        stoichiometric_matrix = np.zeros((len(CHEMICAL_ELEMENTS), len(species)))
        modified_species = [s.replace("@", "").replace("#", "") for s in species]

        for elem_idx, (_, pattern) in enumerate(ELEMENT_PATTERNS.items()):
            for species_idx, spec in enumerate(modified_species):
                if spec in ["SURFACE", "BULK"]:
                    continue
                match = pattern.search(spec)
                if match:
                    multiplier = int(match.group(1)) if match.group(1) else 1
                    stoichiometric_matrix[elem_idx, species_idx] = multiplier

        stoich_filename = getattr(
            self.cfg.output, "stoichiometric_matrix", "stoichiometric_matrix.pt"
        )
        output_path = output_dir / stoich_filename
        torch.save(torch.from_numpy(stoichiometric_matrix.T), output_path)
        print(f"Saved stoichiometric matrix to: {output_path}")
        return stoichiometric_matrix.T

    def run(self, output_dir: Path) -> None:
        """Executes the preprocessing pipeline."""
        print(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")
        species = self._load_species()
        df = self._load_raw_data()

        # rename columns and set radfield minimum
        df = self._clean_dataframe(df)

        # gotta add the initial abundances to each tracer. (forgot to do this during data generation)
        df = self._process_trajectories(df, species)

        # clip abundances to min of 1e-20 bc anything less than that is ODE solver tolerance.
        df = self._clip_abundances(df, species)

        train_tracers, val_tracers = self._get_split_tracers(df)
        train_df = df[df["Model"].isin(train_tracers)]
        val_df = df[df["Model"].isin(val_tracers)]

        train_tensor = self._to_3d_tensor(train_df, train_tracers, species)
        val_tensor = self._to_3d_tensor(val_df, val_tracers, species)

        self._save_data(output_dir, train_tensor, val_tensor)
        self._build_stoichiometric_matrix(species, output_dir)
