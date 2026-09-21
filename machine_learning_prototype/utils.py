# Load up necessary packages. 
import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################ ML #######################################

def weighted_mse_loss(reconstruction, target, weights):
    # Calculate mean squared error separately for each window.
    per_window_loss = (reconstruction - target) ** 2
    per_window_loss = per_window_loss.mean(dim=(1, 2))

    # Weight each window according to its comparison size.
    weighted_loss = (per_window_loss * weights).sum() / weights.sum()

    return(weighted_loss)

def augment_signal(signal, scale_min=0.2, scale_max=5.0):
    # Copy the signal so the original data are not modified.
    augmented = signal.clone()

    # Jointly scale both RBP tracks using a log-uniform distribution.
    log_scale = torch.empty(1).uniform_(
        math.log(scale_min),
        math.log(scale_max)
    )
    scale = torch.exp(log_scale)
    augmented = augmented * scale

    # Randomly flip the global sign of both tracks.
    if torch.rand(1).item() < 0.5:
        augmented = -augmented

    # Randomly swap RBP A and RBP B.
    if torch.rand(1).item() < 0.5:
        augmented = augmented.flip(0)

    # Randomly reverse the nucleotide axis.
    if torch.rand(1).item() < 0.5:
        augmented = augmented.flip(1)

    return(augmented)

class VICRegModel(nn.Module):
    def __init__(self, latent_dim=64, projection_dim=128):
        super().__init__()

        # Reuse the existing autoencoder encoder.
        self.autoencoder = SimpleAutoencoder(latent_dim=latent_dim)

        # Project the encoder representation into the space used by VICReg.
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def encode(self, x):
        return(self.autoencoder.encode(x))

    def forward(self, x):
        latent = self.encode(x)
        projection = self.projector(latent)

        return(projection)


def off_diagonal(x):
    n, m = x.shape
    assert n == m

    return(x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten())


def vicreg_loss(
    z_1,
    z_2,
    sim_coeff=25.0,
    std_coeff=25.0,
    cov_coeff=1.0
):
    # Invariance loss.
    invariance_loss = F.mse_loss(z_1, z_2)

    # Center each branch independently.
    z_1_centered = z_1 - z_1.mean(dim=0)
    z_2_centered = z_2 - z_2.mean(dim=0)

    # Variance loss.
    std_z_1 = torch.sqrt(z_1_centered.var(dim=0) + 0.0001)
    std_z_2 = torch.sqrt(z_2_centered.var(dim=0) + 0.0001)

    variance_loss = (
        torch.mean(F.relu(1 - std_z_1)) / 2 +
        torch.mean(F.relu(1 - std_z_2)) / 2
    )

    # Covariance loss.
    batch_size = z_1.shape[0]
    projection_dim = z_1.shape[1]

    cov_z_1 = (z_1_centered.T @ z_1_centered) / (batch_size - 1)
    cov_z_2 = (z_2_centered.T @ z_2_centered) / (batch_size - 1)

    covariance_loss = (
        off_diagonal(cov_z_1).pow(2).sum() / projection_dim +
        off_diagonal(cov_z_2).pow(2).sum() / projection_dim
    )

    # Combine the three VICReg loss components.
    total_loss = (
        sim_coeff * invariance_loss +
        std_coeff * variance_loss +
        cov_coeff * covariance_loss
    )

    return(total_loss, invariance_loss, variance_loss, covariance_loss)

# Set threads equal to number of available CPUs. 
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder.
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(32 * 75, latent_dim)

        # Decoder.
        self.decoder_fc = nn.Linear(latent_dim, 32 * 75)

        self.deconv1 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.deconv2 = nn.ConvTranspose1d(
            in_channels=16,
            out_channels=2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.encoder_fc(x)

        return(x)

    def decode(self, x):
        x = self.relu(self.decoder_fc(x))

        # Turn the 2400 values back into 32 feature maps × 75 positions.
        x = x.view(-1, 32, 75)

        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)

        return(x)

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)

        return(reconstruction)

################################ PCA ######################################

################################ PCA ######################################

def build_validation_pca(model, val_loader, val_metadata, device):
    model.eval()

    latent_vectors = []
    validation_signals = []

    with torch.no_grad():
        for (batch_signals,) in val_loader:
            batch_signals = batch_signals.to(device)

            latent = model.encode(batch_signals)

            latent_vectors.append(latent.cpu().numpy())
            validation_signals.append(batch_signals.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    validation_signals = np.concatenate(validation_signals, axis=0)

    if len(latent_vectors) != len(val_metadata):
        raise ValueError("Latent vectors and validation metadata are not aligned.")

    n_pcs = 10

    pca = PCA(n_components=n_pcs)
    latent_pca = pca.fit_transform(latent_vectors)

    pca_df = val_metadata.reset_index(drop=True).copy()

    for i in range(n_pcs):
        pca_df[f"PC{i + 1}"] = latent_pca[:, i]

    pca_df["pair_label"] = pca_df["experiment_A"] + " vs " + pca_df["experiment_B"]
    pca_df["pair_key"] = [
        "__".join(sorted((a, b)))
        for a, b in zip(pca_df["experiment_A"], pca_df["experiment_B"])
    ]

    return(pca, pca_df, latent_vectors, validation_signals)


def plot_validation_pca(
    pca_df,
    pca,
    categories=None,
    comparison_ids=None,
    pairs=None,
    color_by="category",
    point_size=10,
    alpha=0.5
):
    plot_df = pca_df.copy()

    if categories is not None:
        plot_df = plot_df[plot_df["category"].isin(categories)]

    if comparison_ids is not None:
        plot_df = plot_df[plot_df["comparison_id"].isin(comparison_ids)]

    if pairs is not None:
        pair_keys = {"__".join(sorted(pair)) for pair in pairs}
        plot_df = plot_df[plot_df["pair_key"].isin(pair_keys)]

    if len(plot_df) == 0:
        print("No validation windows matched those filters.")
        return

    plt.figure(figsize=(7, 6))

    if color_by is None:
        plt.scatter(plot_df["PC1"], plot_df["PC2"], s=point_size, alpha=alpha)
    else:
        for label, group in plot_df.groupby(color_by, sort=False):
            plt.scatter(
                group["PC1"],
                group["PC2"],
                s=point_size,
                alpha=alpha,
                label=label
            )
        plt.legend()

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.show()


def plot_window(validation_signals, pca_df, idx):
    signal = validation_signals[idx]
    row = pca_df.loc[idx]

    plt.figure(figsize=(10, 4))

    plt.plot(
        signal[0],
        label=row["experiment_A"]
    )

    plt.plot(
        signal[1],
        label=row["experiment_B"]
    )

    plt.title(
        f"{row['category']} | PC1={row['PC1']:.2f}, PC2={row['PC2']:.2f}"
    )

    plt.xlabel("Position")
    plt.ylabel("Abundance adjusted fold change")
    plt.legend()
    plt.show()

def build_comparison_pca_summary(pca_df):
    comparison_pca = (
        pca_df
        .groupby(
            [
                "comparison_id",
                "category",
                "experiment_A",
                "experiment_B"
            ],
            as_index=False
        )
        .agg(
            PC1=("PC1", "mean"),
            PC2=("PC2", "mean"),
            n_windows=("PC1", "size")
        )
    )

    comparison_pca["pair_label"] = (
        comparison_pca["experiment_A"]
        + " vs "
        + comparison_pca["experiment_B"]
    )

    return(comparison_pca)


# def build_comparison_latents(latent_vectors, val_metadata):
#     metadata = val_metadata.reset_index(drop=True).copy()

#     if len(latent_vectors) != len(metadata):
#         raise ValueError("Latent vectors and validation metadata are not aligned.")

#     metadata["latent_index"] = np.arange(len(metadata))

#     comparison_rows = []
#     comparison_latents = []

#     for comparison_id, group in metadata.groupby("comparison_id", sort=False):
#         indices = group["latent_index"].to_numpy()

#         mean_latent = latent_vectors[indices].mean(axis=0)

#         comparison_latents.append(mean_latent)
#         comparison_rows.append({
#             "comparison_id": comparison_id,
#             "category": group["category"].iloc[0],
#             "experiment_A": group["experiment_A"].iloc[0],
#             "experiment_B": group["experiment_B"].iloc[0],
#             "n_windows": len(group)
#         })

#     comparison_latents = np.stack(comparison_latents)
#     comparison_metadata = pd.DataFrame(comparison_rows)

#     comparison_metadata["pair_label"] = (
#         comparison_metadata["experiment_A"] + " vs " +
#         comparison_metadata["experiment_B"]
#     )

#     return(comparison_latents, comparison_metadata)

# def build_validation_pca(model, val_loader, val_metadata, device):
#     model.eval()

#     latent_vectors = []
#     validation_signals = []

#     with torch.no_grad():
#         for batch_signals, batch_weights in val_loader:
#             batch_signals = batch_signals.to(device)

#             latent = model.encode(batch_signals)

#             latent_vectors.append(latent.cpu().numpy())
#             validation_signals.append(batch_signals.cpu().numpy())

#     latent_vectors = np.concatenate(latent_vectors, axis=0)
#     validation_signals = np.concatenate(validation_signals, axis=0)

#     if len(latent_vectors) != len(val_metadata):
#         raise ValueError("Latent vectors and validation metadata are not aligned.")

#     n_pcs = 10

#     pca = PCA(n_components=n_pcs)
#     latent_pca = pca.fit_transform(latent_vectors)

#     pca_df = val_metadata.reset_index(drop=True).copy()

#     for i in range(n_pcs):
#         pca_df[f"PC{i + 1}"] = latent_pca[:, i]

#     pca_df["pair_label"] = pca_df["experiment_A"] + " vs " + pca_df["experiment_B"]
#     pca_df["pair_key"] = [
#         "__".join(sorted((a, b)))
#         for a, b in zip(pca_df["experiment_A"], pca_df["experiment_B"])
#     ]

#     return pca, pca_df, latent_vectors, validation_signals


# def plot_validation_pca(pca_df, pca, categories=None, comparison_ids=None, pairs=None,
#                         color_by="category", point_size=10, alpha=0.5):
#     plot_df = pca_df.copy()

#     if categories is not None:
#         plot_df = plot_df[plot_df["category"].isin(categories)]

#     if comparison_ids is not None:
#         plot_df = plot_df[plot_df["comparison_id"].isin(comparison_ids)]

#     if pairs is not None:
#         pair_keys = {"__".join(sorted(pair)) for pair in pairs}
#         plot_df = plot_df[plot_df["pair_key"].isin(pair_keys)]

#     if len(plot_df) == 0:
#         print("No validation windows matched those filters.")
#         return

#     plt.figure(figsize=(7, 6))

#     if color_by is None:
#         plt.scatter(plot_df["PC1"], plot_df["PC2"], s=point_size, alpha=alpha)
#     else:
#         for label, group in plot_df.groupby(color_by, sort=False):
#             plt.scatter(group["PC1"], group["PC2"], s=point_size, alpha=alpha, label=label)
#         plt.legend()

#     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
#     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
#     plt.show()

# def plot_window_reconstruction(validation_signals, model, pca_df, idx):
#     original = validation_signals[idx]

#     # Use the model's device rather than the module-level default. This keeps
#     # analysis reproducible when a saved model is loaded on a different device.
#     model_device = next(model.parameters()).device
#     x = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(model_device)

#     model.eval()
#     with torch.no_grad():
#         reconstruction = model(x)[0].cpu().numpy()

#     row = pca_df.loc[idx]

#     plt.figure(figsize=(10, 4))
#     plt.plot(original[0], label=f"{row['experiment_A']} original")
#     plt.plot(reconstruction[0], linestyle="--", label=f"{row['experiment_A']} reconstruction")
#     plt.plot(original[1], label=f"{row['experiment_B']} original")
#     plt.plot(reconstruction[1], linestyle="--", label=f"{row['experiment_B']} reconstruction")

#     plt.title(
#         f"{row['category']} | PC1={row['PC1']:.2f}, PC2={row['PC2']:.2f}"
#     )

#     plt.xlabel("Position")
#     plt.ylabel("Abundance adjusted fold change")
#     plt.legend()
#     plt.show()

# def build_comparison_latents(latent_vectors, val_metadata):
#     metadata = val_metadata.reset_index(drop=True).copy()

#     if len(latent_vectors) != len(metadata):
#         raise ValueError("Latent vectors and validation metadata are not aligned.")

#     metadata["latent_index"] = np.arange(len(metadata))

#     comparison_rows = []
#     comparison_latents = []

#     for comparison_id, group in metadata.groupby("comparison_id", sort=False):
#         indices = group["latent_index"].to_numpy()

#         mean_latent = latent_vectors[indices].mean(axis=0)

#         comparison_latents.append(mean_latent)
#         comparison_rows.append({
#             "comparison_id": comparison_id,
#             "category": group["category"].iloc[0],
#             "experiment_A": group["experiment_A"].iloc[0],
#             "experiment_B": group["experiment_B"].iloc[0],
#             "n_windows": len(group)
#         })

#     comparison_latents = np.stack(comparison_latents)
#     comparison_metadata = pd.DataFrame(comparison_rows)

#     comparison_metadata["pair_label"] = (
#         comparison_metadata["experiment_A"] + " vs " +
#         comparison_metadata["experiment_B"]
#     )

#     return comparison_latents, comparison_metadata
