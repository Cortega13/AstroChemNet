import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Emulator(nn.Module):
    def __init__(self, input_dim=18, output_dim=14, hidden_dim=32, dropout=0.0):
        super(Emulator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, phys, latents):
        B, T, P = phys.shape
        L = latents.shape[1]
        outputs = torch.empty(B, T, L, device=latents.device, dtype=latents.dtype)

        for t in range(T):
            current_phys = phys[:, t, :]  # [B, P]
            input = torch.cat([current_phys, latents], dim=1)  # [B, P+L]
            latents = latents + self.net(input)
            outputs[:, t, :] = latents

        return outputs
    

def load_emulator(GeneralConfig, EMConfig, inference=False):
    emulator = Emulator(
        input_dim=EMConfig.input_dim,
        output_dim=EMConfig.output_dim,
        hidden_dim=EMConfig.hidden_dim
    ).to(GeneralConfig.device)
    if os.path.exists(EMConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        emulator.load_state_dict(torch.load(EMConfig.pretrained_model_path))
    if inference:
        emulator.eval()
        for param in emulator.parameters():
            param.requires_grad = False
    return emulator