"""Utility functions for data processing and transformation."""

from typing import List

import numpy as np
import torch


def rename_columns(columns: List[str]) -> List[str]:
    """Rename column names containing chemical species using substring replacement."""
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


def convertUCLCHEMbaseAvtoAv(physical_parameters: np.ndarray) -> None:
    """Convert baseAv to Av using UCLCHEM internal conversion formula."""
    baseAv_idx = 2
    density_idx = 0
    multiplier = 0.0000964375
    additive = np.multiply(multiplier, physical_parameters[:, density_idx])
    np.add(
        physical_parameters[:, baseAv_idx],
        additive,
        out=physical_parameters[:, baseAv_idx],
    )


def reconstruct_emulated_outputs(
    encoded_inputs: torch.Tensor,
    emulated_outputs: torch.Tensor,
    num_physical_parameters: int,
) -> torch.Tensor:
    """Add time and physical parameter columns to the latent components."""
    reconstructed_emulated_outputs = torch.cat(
        (encoded_inputs[:, : 1 + num_physical_parameters], emulated_outputs), dim=1
    )
    return reconstructed_emulated_outputs
