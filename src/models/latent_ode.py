"""Latent ODE model for chemical abundance evolution prediction."""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint

from src.configs.datasets import DatasetConfig
from src.configs.latent_ode import LatentODEConfig


class ControlledLatentODEFunc(nn.Module):
    """Latent dynamics function with piecewise-constant physical controls."""

    def __init__(
        self,
        latent_dim: int,
        phys_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
    ) -> None:
        """Build latent dynamics network."""
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim + phys_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self.current_control: Optional[torch.Tensor] = None

        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.bias)
            final_layer.weight.data.mul_(1e-2)

    def set_control(self, control: torch.Tensor) -> None:
        """Set the current piecewise-constant control value."""
        self.current_control = control

    def forward(self, _t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return latent derivative for the current control interval."""
        if self.current_control is None:
            raise RuntimeError("Control must be set before solving the latent ODE.")
        inputs = torch.cat([z, self.current_control], dim=1)
        return self.net(inputs)


class LatentODE(nn.Module):
    """Piecewise-controlled latent ODE rollout model."""

    def __init__(
        self,
        latent_dim: int,
        phys_dim: int,
        hidden_dim: int = 192,
        num_hidden_layers: int = 2,
        method: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-6,
        solver_substeps: int = 4,
    ) -> None:
        """Initialize model and solver settings."""
        super().__init__()
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.solver_substeps = solver_substeps
        self.func = ControlledLatentODEFunc(
            latent_dim=latent_dim,
            phys_dim=phys_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.register_buffer(
            "integration_time",
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            persistent=False,
        )

    def _solve_interval(
        self,
        z0: torch.Tensor,
        phys: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one interval, grouping equal time deltas for efficient batching."""
        if z0.numel() == 0:
            return z0

        next_z = torch.empty_like(z0)
        unique_dt, inverse = torch.unique(delta_t.detach(), sorted=True, return_inverse=True)

        for idx, dt in enumerate(unique_dt):
            mask = inverse == idx
            group_z0 = z0[mask]
            group_phys = phys[mask]
            self.func.set_control(group_phys)

            t_eval = self.integration_time.to(device=z0.device, dtype=z0.dtype) * dt
            options = None
            if self.method == "rk4":
                step_size = float(dt.item()) / max(self.solver_substeps, 1)
                options = {"step_size": step_size}

            solution = odeint(
                self.func,
                group_z0,
                t_eval,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                options=options,
            )
            next_z[mask] = solution[-1]

        return next_z

    def forward(
        self,
        delta_t: torch.Tensor,
        phys: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate latent state through all observation intervals."""
        batch_size, timesteps, _ = phys.shape
        outputs = torch.empty(
            batch_size,
            timesteps,
            latents.shape[1],
            device=latents.device,
            dtype=latents.dtype,
        )
        current = latents

        for timestep in range(timesteps):
            current = self._solve_interval(
                current,
                phys[:, timestep, :],
                delta_t[:, timestep],
            )
            outputs[:, timestep, :] = current

        return outputs


def load_latent_ode(
    model_class: type[LatentODE],
    general_config: DatasetConfig,
    ode_config: LatentODEConfig,
    inference: bool = False,
) -> LatentODE:
    """Load latent ODE model with optional pretrained weights."""
    model = model_class(
        latent_dim=ode_config.output_dim,
        phys_dim=general_config.num_phys,
        hidden_dim=ode_config.hidden_dim,
        num_hidden_layers=ode_config.num_hidden_layers,
        method=ode_config.method,
        rtol=ode_config.rtol,
        atol=ode_config.atol,
        solver_substeps=ode_config.solver_substeps,
    ).to(general_config.device)

    if os.path.exists(ode_config.pretrained_model_path):
        print("Loading Pretrained Model")
        model.load_state_dict(
            torch.load(
                ode_config.pretrained_model_path, map_location=torch.device("cpu")
            )
        )

    if inference:
        print("Setting LatentODE to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model


def save_base_dt(path: str, base_dt: float) -> None:
    """Persist the dataset base time interval for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"base_dt": float(base_dt)}, f, indent=2)
