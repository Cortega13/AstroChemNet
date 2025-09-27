import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim=333,
        latent_dim=12,
        hidden_dims=(320, 160),
        noise=0.1,
        dropout=0.0,
    ):
        super(Autoencoder, self).__init__()

        self.encoder_fc1 = nn.Linear(input_dim, hidden_dims[0], bias=False)
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.encoder_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.encoder_bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.encoder_fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self.encoder_norm3 = nn.BatchNorm1d(latent_dim)

        self.decoder_bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.decoder_bn2 = nn.BatchNorm1d(hidden_dims[0])

        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dims[1]))
        self.decoder_bias2 = nn.Parameter(torch.zeros(hidden_dims[0]))
        self.decoder_bias3 = nn.Parameter(torch.zeros(input_dim))

        self.activation = nn.GELU()
        self.final_activation = nn.Sigmoid()  # For 0-1 bounded output
        self.dropout = nn.Dropout(dropout)
        self.noise = noise

    def encode(self, x):
        x = self.activation(self.encoder_bn1(self.encoder_fc1(x)))
        x = self.activation(self.encoder_bn2(self.encoder_fc2(x)))
        x = self.dropout(x)
        z = self.activation(self.encoder_norm3(self.encoder_fc3(x)))
        return z

    def decode(self, z):
        z = F.linear(z, self.encoder_fc3.weight.t()) + self.decoder_bias1
        z = self.activation(self.decoder_bn1(z))

        z = F.linear(z, self.encoder_fc2.weight.t()) + self.decoder_bias2
        z = self.activation(self.decoder_bn2(z))
        z = self.dropout(z)

        x_reconstructed = F.linear(z, self.encoder_fc1.weight.t()) + self.decoder_bias3
        x_reconstructed = self.final_activation(x_reconstructed)
        return x_reconstructed

    def forward(self, x):
        z = self.encode(x)
        if self.training and self.noise > 0:
            noise = torch.randn_like(z) * self.noise
            z += noise
        x_reconstructed = self.decode(z)
        return x_reconstructed


def load_autoencoder(
    Autoencoder: Autoencoder, GeneralConfig, AEConfig, inference=False
):
    autoencoder = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dims=AEConfig.hidden_dims,
        noise=AEConfig.noise,
        dropout=AEConfig.dropout,
    ).to(GeneralConfig.device)
    if os.path.exists(AEConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        autoencoder.load_state_dict(
            torch.load(AEConfig.pretrained_model_path, map_location=torch.device("cpu"))
        )

    if inference:
        print("Setting Autoencoder to Inference Mode")
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False

    return autoencoder
