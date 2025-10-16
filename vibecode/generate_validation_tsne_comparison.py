import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(".."))

import AstroChemNet.data_loading as dl
from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from configs.general import GeneralConfig
from nn_architectures.autoencoder import Autoencoder, load_autoencoder
from nn_architectures.emulator import Emulator, load_emulator
from src.AstroChemNet.data_processing import Processing
from src.AstroChemNet.inference import Inference

# Load configs
general_config = GeneralConfig()
ae_config = AEConfig()
em_config = EMConfig()

# Load models
autoencoder = load_autoencoder(Autoencoder, general_config, ae_config, inference=True)
emulator = Emulator(
    input_dim=em_config.input_dim,
    output_dim=em_config.output_dim,
    hidden_dim=180,  # Match saved model
).to(general_config.device)
if os.path.exists(em_config.pretrained_model_path):
    print("Loading Pretrained Model")
    emulator.load_state_dict(
        torch.load(em_config.pretrained_model_path, map_location=torch.device("cpu"))
    )
emulator.eval()
for param in emulator.parameters():
    param.requires_grad = False

# Load processing and inference
processing = Processing(general_config, ae_config)
inference = Inference(general_config, processing, autoencoder, emulator)

# Load data
training_np, validation_np = dl.load_datasets(general_config, em_config.columns)

# Load validation indices
val_indices = np.loadtxt("../scripts/preprocessing/val_indices.csv", dtype=int)

# Filter validation data to only validation models
val_models = set(val_indices)
val_mask = np.isin(validation_np[:, 1], val_indices)  # column 1 is model_id
validation_filtered = validation_np[val_mask]

# Group by model
unique_models = np.unique(validation_filtered[:, 1])

predicted_trajectories = []
ground_truth_trajectories = []
log_densities = []

for model in unique_models:
    model_mask = validation_filtered[:, 1] == model
    model_data = validation_filtered[model_mask]

    # Sort by time (column 2 is time)
    time_col = 2
    model_data = model_data[np.argsort(model_data[:, time_col])]

    # Ground truth abundances
    num_metadata = general_config.num_metadata  # index, model, time
    num_phys = general_config.num_phys
    abundances_cols = slice(num_metadata + num_phys, None)
    gt_abundances = model_data[:, abundances_cols]
    log_gt = np.log10(gt_abundances + 1e-20).flatten()
    ground_truth_trajectories.append(log_gt)

    # Final density for coloring (density is column 3, after index, model, time)
    final_density = model_data[-1, 3]
    log_densities.append(np.log10(final_density))

    # Predict sequence
    initial_abundances = model_data[0, abundances_cols]
    phys_params = model_data[:, num_metadata : num_metadata + num_phys]

    # Scale phys params
    phys_scaled = phys_params.copy()
    processing.physical_parameter_scaling(phys_scaled)

    # Start with initial latents
    latents = inference.encode(initial_abundances.reshape(1, -1)).squeeze()

    predicted_abundances = [initial_abundances]

    for t in range(1, len(model_data)):
        phys_t = phys_scaled[t].reshape(1, 1, -1)  # Add time dimension
        next_latents = inference.latent_emulate(
            phys_t, latents.reshape(1, -1)
        ).squeeze()
        next_abundances = inference.decode(next_latents.reshape(1, -1)).squeeze()
        predicted_abundances.append(next_abundances.cpu().numpy())
        latents = next_latents

    pred_abundances = np.array(predicted_abundances)
    log_pred = np.log10(pred_abundances + 1e-20).flatten()
    predicted_trajectories.append(log_pred)

# Convert to arrays
predicted_data = np.array(predicted_trajectories)
ground_truth_data = np.array(ground_truth_trajectories)
log_densities = np.array(log_densities)

# Load original t-SNE data to get the training/validation trajectories
original_tsne = np.loadtxt("tsne_data.csv", delimiter=",", skiprows=1)
tracer_indices_orig = original_tsne[:, 0]
tsne1_orig = original_tsne[:, 1]
tsne2_orig = original_tsne[:, 2]
log_densities_orig = original_tsne[:, 3]

# Load train and val indices
train_indices = np.loadtxt("../scripts/preprocessing/train_indices.csv", dtype=int)
val_indices = np.loadtxt("../scripts/preprocessing/val_indices.csv", dtype=int)

# Get the original trajectories for training and validation
train_mask = np.isin(tracer_indices_orig, train_indices)
val_mask = np.isin(tracer_indices_orig, val_indices)

# Load the full dataset to get trajectories
training_np, validation_np = dl.load_datasets(general_config, em_config.columns)

# Get training trajectories
train_trajectories = []
for idx in train_indices[:100]:  # Limit to first 100 for speed
    model_mask = training_np[:, 1] == idx
    model_data = training_np[model_mask]
    if len(model_data) > 0:
        model_data = model_data[np.argsort(model_data[:, 2])]  # sort by time
        abundances = model_data[:, -general_config.num_species :]
        log_abund = np.log10(abundances + 1e-20).flatten()
        train_trajectories.append(log_abund)

# Get validation GT trajectories
val_gt_trajectories = []
for idx in val_indices[:50]:  # Limit to first 50 for speed
    model_mask = validation_np[:, 1] == idx
    model_data = validation_np[model_mask]
    if len(model_data) > 0:
        model_data = model_data[np.argsort(model_data[:, 2])]  # sort by time
        abundances = model_data[:, -general_config.num_species :]
        log_abund = np.log10(abundances + 1e-20).flatten()
        val_gt_trajectories.append(log_abund)

# Fit PCA and t-SNE on training + validation GT + predicted (combined)
combined_fit_data = np.vstack(
    [np.array(train_trajectories + val_gt_trajectories), predicted_data]
)
pca_combined = PCA(n_components=50)
pca_reduced_combined = pca_combined.fit_transform(combined_fit_data)

# Use lower perplexity to avoid the error
n_samples = len(combined_fit_data)
perplexity = min(30, n_samples - 1)  # Ensure perplexity < n_samples
tsne_combined = TSNE(n_components=2, random_state=42, perplexity=perplexity)
embedded_combined = tsne_combined.fit_transform(pca_reduced_combined)

# Split back
n_train = len(train_trajectories)
n_val_gt = len(val_gt_trajectories)
embedded_train = embedded_combined[:n_train]
embedded_val_gt = embedded_combined[n_train : n_train + n_val_gt]
embedded_pred = embedded_combined[n_train + n_val_gt :]

# Save data
pred_data_save = np.column_stack(
    (unique_models, embedded_pred[:, 0], embedded_pred[:, 1], log_densities)
)
np.savetxt(
    "tsne_predicted_validation.csv",
    pred_data_save,
    delimiter=",",
    header="tracer_index,tsne1,tsne2,log_density",
    comments="",
)

# Skip saving GT data for now since we have dimension mismatch
# The plot is the main output

# Load original t-SNE data for training and validation GT
original_tsne = np.loadtxt("tsne_data.csv", delimiter=",", skiprows=1)
tracer_indices_orig = original_tsne[:, 0]
tsne1_orig = original_tsne[:, 1]
tsne2_orig = original_tsne[:, 2]
log_densities_orig = original_tsne[:, 3]

# Load train and val indices
train_indices = np.loadtxt("../scripts/preprocessing/train_indices.csv", dtype=int)
val_indices = np.loadtxt("../scripts/preprocessing/val_indices.csv", dtype=int)

# Filter masks
train_mask = np.isin(tracer_indices_orig, train_indices)
val_mask = np.isin(tracer_indices_orig, val_indices)

# Plot combined
os.makedirs("../plots/trajectories", exist_ok=True)

plt.figure(figsize=(10, 8))

# Training data: red
plt.scatter(embedded_train[:, 0], embedded_train[:, 1], c="red", s=1, label="Training")

# Validation GT: blue
plt.scatter(
    embedded_val_gt[:, 0], embedded_val_gt[:, 1], c="blue", s=1, label="Validation GT"
)

# Predicted: purple
plt.scatter(
    embedded_pred[:, 0], embedded_pred[:, 1], c="purple", s=1, label="Predicted"
)

plt.title("t-SNE: Training (Red), Validation GT (Blue), Predicted (Purple)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/tsne_training_val_pred_overlay_fitted.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Validation t-SNE comparison generated.")
