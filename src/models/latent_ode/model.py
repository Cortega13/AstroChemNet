"""Latent ODE model for chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn
import torchode as to

from src.datasets import DatasetConfig
from src.models.latent_ode.config import LatentODEConfig


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
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.bias)
            final_layer.weight.data.mul_(1e-2)

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        args: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Return latent derivative for the current control interval."""
        controls, t_eval = args
        control_index = torch.searchsorted(
            t_eval.contiguous(),
            t.unsqueeze(-1),
            right=True,
        )
        control_index = torch.clamp(control_index, max=controls.shape[1] - 1)
        batch_index = torch.arange(z.shape[0], device=z.device)
        control = controls[batch_index, control_index.squeeze(-1)]
        inputs = torch.cat([z, control], dim=1)
        return self.net(inputs)


class LatentODE(nn.Module):
    """Piecewise-controlled latent ODE rollout model."""

    def __init__(
        self,
        latent_dim: int,
        phys_dim: int,
        hidden_dim: int = 80,
        num_hidden_layers: int = 2,
        method: str = "tsit5",
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ) -> None:
        """Initialize model and solver settings."""
        super().__init__()
        self.func = ControlledLatentODEFunc(
            latent_dim=latent_dim,
            phys_dim=phys_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        term = to.ODETerm(self.func, with_args=True, with_stats=False)
        method_map = {
            "tsit5": to.Tsit5,
            "dopri5": to.Dopri5,
        }
        if method not in method_map:
            raise ValueError(f"Unsupported latent ODE method: {method}")
        step_method = method_map[method](term)
        controller = to.IntegralController(
            atol=atol,
            rtol=rtol,
            term=term,
        )
        self.solver = to.AutoDiffAdjoint(step_method, controller)  # type:ignore

    def forward(
        self,
        delta_t: torch.Tensor,
        phys: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate latent state through all observation intervals."""
        t_eval = torch.cumsum(delta_t, dim=1)
        problem = to.InitialValueProblem(
            y0=latents,  # type:ignore
            t_start=torch.zeros_like(delta_t[:, 0]),  # type:ignore
            t_end=t_eval[:, -1],  # type:ignore
            t_eval=t_eval,  # type:ignore
        )
        solution = self.solver.solve(problem, args=(phys, t_eval))
        return torch.clamp(solution.ys, 0.0, 1.0)


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
    ).to(general_config.device)

    if os.path.exists(ode_config.pretrained_model_path):
        print("Loading Pretrained Model")
        model.load_state_dict(
            torch.load(
                ode_config.pretrained_model_path,
                map_location=torch.device("cpu"),
            )
        )

    if inference:
        print("Setting LatentODE to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model
