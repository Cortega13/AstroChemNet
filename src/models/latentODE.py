import numpy as np
import torch
import torch.nn as nn
import torchode as tode


class A(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate=0.1):
        super(A, self).__init__()

        self.z_dim = z_dim
        self.dropout_rate = dropout_rate

        hidden_dim1 = z_dim
        out_dim = z_dim**2
        hidden_dim2 = out_dim // 2

        self.layer_in = nn.Linear(input_dim, hidden_dim1)
        self.layer_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_out = nn.Linear(hidden_dim2, out_dim)

        nn.init.kaiming_normal_(self.layer_in.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_hidden.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_out.weight, a=0.2)

        scale = 0.1
        bias = torch.diag(-torch.ones(z_dim) * scale)
        self.layer_out.bias.data = bias.flatten()

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.dropout(h)  # Apply dropout after activation

        h = self.LeakyReLU(self.layer_hidden(h))
        h = self.dropout(h)  # Apply dropout after activation

        h = self.LeakyReLU(self.layer_out(h))
        # No dropout on the final output as it's reconstructing the matrix A

        return h.reshape(self.z_dim, self.z_dim)


class B(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate=0.1):
        super(B, self).__init__()

        self.z_dim = z_dim
        self.dropout_rate = dropout_rate

        hidden_dim1 = z_dim
        out_dim = z_dim**3
        hidden_dim2 = int(np.sqrt(out_dim))
        hidden_dim3 = out_dim // 2

        self.layer_in = nn.Linear(input_dim, hidden_dim1)
        self.layer_hidden1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_hidden2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.layer_out = nn.Linear(hidden_dim3, out_dim)

        nn.init.kaiming_normal_(self.layer_in.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_hidden1.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_hidden2.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_out.weight, a=0.2)
        self.layer_out.weight.data *= 0.1

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.dropout(h)  # Apply dropout after activation

        h = self.LeakyReLU(self.layer_hidden1(h))
        h = self.dropout(h)  # Apply dropout after activation

        h = self.LeakyReLU(self.layer_hidden2(h))
        h = self.dropout(h)  # Apply dropout after activation

        h = self.LeakyReLU(self.layer_out(h))
        # No dropout on the final output tensor

        return h.reshape(self.z_dim, self.z_dim, self.z_dim)


class LatentODEFunction(nn.Module):
    """MACE latent ODE two body equations form."""

    def __init__(self, z_dim, phys_param_dim):
        super(LatentODEFunction, self).__init__()
        self.z_dim = z_dim

        # Base trainable dynamics
        self.C = nn.Parameter(torch.zeros(z_dim))
        A_init = torch.randn(z_dim, z_dim) * 0.1
        U, _, V = torch.svd(A_init)
        D = torch.diag(torch.rand(z_dim) * -0.5)
        A_init = U @ D @ V.t()
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(torch.randn(z_dim, z_dim, z_dim) * 0.01)

        # Physical parameter modulation
        self.phys_modulation = nn.Sequential(
            nn.Linear(phys_param_dim, 64),
            nn.Tanh(),
            nn.Linear(64, z_dim),
            nn.Sigmoid(),  # Scale between 0-1
        )

        # Rate scaling based on physical parameters (e.g., temperature)
        self.rate_scaling = nn.Sequential(
            nn.Linear(phys_param_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Positive scaling
        )

    def forward(self, t, z, p):
        # Get physical parameter effects
        mod_factors = self.phys_modulation(p)
        rate_scale = self.rate_scaling(p)

        # Apply standard dynamics with physical modulation
        linear_term = torch.einsum("ij, bj -> bi", self.A, z * mod_factors)
        quad_term = torch.einsum("ijk, bj, bk -> bi", self.B, z, z)

        # Scale by global rate factor (e.g., temperature effect)
        return rate_scale * (self.C + linear_term + quad_term)


class LatentODE(nn.Module):
    """General latent ODE form."""

    def __init__(
        self, input_dim, hidden_dim, latent_dim, output_dim, jit_solver, dropout=0.0
    ):
        super(LatentODE, self).__init__()

        self.encoder = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # Encoder Hidden Layer
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),
        )
        self.jit_solver = jit_solver

        self.decoder = nn.Sequential(
            # Decoder Hidden Layer
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            # Output Layer
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Data is normalized between 0 & 1.
        )

    def forward(self, t, p, x):
        """Encode abundances, solve IVP ODE to get future latent abundances, then decode and return them."""
        z = self.encoder(x)

        problem = tode.InitialValueProblem(
            y0=z,
            t_eval=t,
        )

        z = self.jit_solver.solve(problem, args=p).ys

        z_reshaped = z.reshape(z.size(0) * z.size(1), z.size(2))

        x_recon_reshaped = self.decoder(z_reshaped)

        x_recon = x_recon_reshaped.view(z.size(0), z.size(1), -1)

        return x_recon
