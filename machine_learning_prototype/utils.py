# Load up necessary packages. 
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
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

def build_validation_pca(model, val_loader, val_metadata, device):
    model.eval()

    latent_vectors = []
    validation_signals = []

    with torch.no_grad():
        for batch_signals, batch_weights in val_loader:
            batch_signals = batch_signals.to(device)

            latent = model.encode(batch_signals)

            latent_vectors.append(latent.cpu().numpy())
            validation_signals.append(batch_signals.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    validation_signals = np.concatenate(validation_signals, axis=0)

    if len(latent_vectors) != len(val_metadata):
        raise ValueError("Latent vectors and validation metadata are not aligned.")

    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)

    pca_df = val_metadata.reset_index(drop=True).copy()
    pca_df["PC1"] = latent_pca[:, 0]
    pca_df["PC2"] = latent_pca[:, 1]
    pca_df["pair_label"] = pca_df["experiment_A"] + " vs " + pca_df["experiment_B"]
    pca_df["pair_key"] = [
        "__".join(sorted((a, b)))
        for a, b in zip(pca_df["experiment_A"], pca_df["experiment_B"])
    ]

    return pca, pca_df, latent_vectors, validation_signals


def plot_validation_pca(pca_df, pca, categories=None, comparison_ids=None, pairs=None,
                        color_by="category", point_size=10, alpha=0.5):
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
            plt.scatter(group["PC1"], group["PC2"], s=point_size, alpha=alpha, label=label)
        plt.legend()

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.show()

def plot_window_reconstruction(validation_signals, model, pca_df, idx):
    original = validation_signals[idx]

    x = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        reconstruction = model(x)[0].cpu().numpy()

    row = pca_df.loc[idx]

    plt.figure(figsize=(10, 4))
    plt.plot(original[0], label=f"{row['experiment_A']} original")
    plt.plot(reconstruction[0], linestyle="--", label=f"{row['experiment_A']} reconstruction")
    plt.plot(original[1], label=f"{row['experiment_B']} original")
    plt.plot(reconstruction[1], linestyle="--", label=f"{row['experiment_B']} reconstruction")

    plt.title(
        f"{row['category']} | PC1={row['PC1']:.2f}, PC2={row['PC2']:.2f}"
    )

    plt.xlabel("Position")
    plt.ylabel("log1p(RPM)")
    plt.legend()
    plt.show()

def build_comparison_latents(latent_vectors, val_metadata):
    metadata = val_metadata.reset_index(drop=True).copy()

    if len(latent_vectors) != len(metadata):
        raise ValueError("Latent vectors and validation metadata are not aligned.")

    metadata["latent_index"] = np.arange(len(metadata))

    comparison_rows = []
    comparison_latents = []

    for comparison_id, group in metadata.groupby("comparison_id", sort=False):
        indices = group["latent_index"].to_numpy()

        mean_latent = latent_vectors[indices].mean(axis=0)

        comparison_latents.append(mean_latent)
        comparison_rows.append({
            "comparison_id": comparison_id,
            "category": group["category"].iloc[0],
            "experiment_A": group["experiment_A"].iloc[0],
            "experiment_B": group["experiment_B"].iloc[0],
            "n_windows": len(group)
        })

    comparison_latents = np.stack(comparison_latents)
    comparison_metadata = pd.DataFrame(comparison_rows)

    comparison_metadata["pair_label"] = (
        comparison_metadata["experiment_A"] + " vs " +
        comparison_metadata["experiment_B"]
    )

    return comparison_latents, comparison_metadata