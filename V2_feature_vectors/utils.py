from scipy.stats import pearsonr
import os
import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def calculate_rpm_pseudocount(library_sizes):
    library_sizes = np.atleast_1d(library_sizes).astype(float)
    return np.mean(1e6 / library_sizes)

def abundance_weighted_enrichment(ip, in_signal, ip_pseudocount, in_pseudocount,
                                  mean_core_signal, p=2):
    k = p * mean_core_signal

    log_enrichment = np.log(
        (ip + ip_pseudocount) /
        (in_signal + in_pseudocount)
    )
    log_enrichment = np.maximum(log_enrichment, 0)

    denominator = ip + k
    abundance_weight = np.divide(
        ip,
        denominator,
        out=np.zeros_like(ip, dtype=float),
        where=denominator > 0
    )

    return log_enrichment * abundance_weight

def C17(signal_A, signal_B, max_lag=150, core_size=300):
    lags = np.arange(-max_lag, max_lag + 1)

    context = (signal_A.shape[2] - core_size) // 2
    core_start = context
    core_end = context + core_size

    if max_lag > context:
        raise ValueError("max_lag exceeds available context.")

    # Central 300-nt analytical window.
    A_core = signal_A[:, :, core_start:core_end]
    B_core = signal_B[:, :, core_start:core_end]

    # Calculate concordance penalty once from the unshifted core.
    mean_A = A_core.mean(axis=2)
    mean_B = B_core.mean(axis=2)
    var_A = A_core.var(axis=2)
    var_B = B_core.var(axis=2)

    cb_denominator = (
        var_A[:, :, None]
        + var_B[:, None, :]
        + (mean_A[:, :, None] - mean_B[:, None, :]) ** 2
    )

    cb = np.full(cb_denominator.shape, np.nan, dtype=np.float32)
    np.divide(
        2 * np.sqrt(var_A[:, :, None] * var_B[:, None, :]),
        cb_denominator,
        out=cb,
        where=cb_denominator > 0
    )

    correlations = np.full(
        (len(signal_A), signal_A.shape[1], signal_B.shape[1], len(lags)),
        np.nan,
        dtype=np.float32
    )

    # A stays anchored to the 300-nt core.
    # B always contributes a 300-nt slice, shifted through its context.
    for i, lag in enumerate(lags):
        A = A_core
        B = signal_B[:, :, core_start + lag:core_end + lag]

        A_centered = A - A.mean(axis=2, keepdims=True)
        B_centered = B - B.mean(axis=2, keepdims=True)

        numerator = np.einsum("wap,wbp->wab", A_centered, B_centered)
        sumsq_A = np.sum(A_centered ** 2, axis=2)
        sumsq_B = np.sum(B_centered ** 2, axis=2)
        denominator = np.sqrt(sumsq_A[:, :, None] * sumsq_B[:, None, :])

        rho = np.zeros(denominator.shape, dtype=np.float32)
        np.divide(numerator, denominator, out=rho, where=denominator > 0)

        correlations[:, :, :, i] = rho * cb

        correlations[:, :, :, i] = np.where(
            (cb_denominator > 0) & (cb == 0),
            0,
            correlations[:, :, :, i]
        )

    return correlations, lags
    
def calculate_fixed_ccc_cross_correlation(ip_A, ip_B, library_sizes_A, library_sizes_B, max_lag=30):
    lags = np.arange(-max_lag, max_lag + 1)

    rpm_A = ip_A * (1e6 / library_sizes_A[None, :, None])
    rpm_B = ip_B * (1e6 / library_sizes_B[None, :, None])

    # Calculate concordance penalty once from the full, unshifted 300-nt windows.
    mean_A = rpm_A.mean(axis=2)
    mean_B = rpm_B.mean(axis=2)
    var_A = rpm_A.var(axis=2)
    var_B = rpm_B.var(axis=2)

    cb_denominator = (
        var_A[:, :, None]
        + var_B[:, None, :]
        + (mean_A[:, :, None] - mean_B[:, None, :]) ** 2
    )

    cb = np.full(cb_denominator.shape, np.nan, dtype=np.float32)
    np.divide(
        2 * np.sqrt(var_A[:, :, None] * var_B[:, None, :]),
        cb_denominator,
        out=cb,
        where=cb_denominator > 0
    )

    correlations = np.full(
        (len(ip_A), ip_A.shape[1], ip_B.shape[1], len(lags)),
        np.nan,
        dtype=np.float32
    )

    # Only the Pearson-like shape term changes with lag.
    for i, lag in enumerate(lags):
        if lag > 0:
            A = rpm_A[:, :, :-lag]
            B = rpm_B[:, :, lag:]
        elif lag < 0:
            A = rpm_A[:, :, -lag:]
            B = rpm_B[:, :, :lag]
        else:
            A = rpm_A
            B = rpm_B

        A_centered = A - A.mean(axis=2, keepdims=True)
        B_centered = B - B.mean(axis=2, keepdims=True)

        numerator = np.einsum("wap,wbp->wab", A_centered, B_centered)
        sumsq_A = np.sum(A_centered ** 2, axis=2)
        sumsq_B = np.sum(B_centered ** 2, axis=2)
        denominator = np.sqrt(sumsq_A[:, :, None] * sumsq_B[:, None, :])

        rho = np.full(denominator.shape, np.nan, dtype=np.float32)
        np.divide(numerator, denominator, out=rho, where=denominator > 0)

        correlations[:, :, :, i] = rho * cb

        # Preserve ordinary CCC behavior when one full track is flat:
        # one flat + one variable = 0; both flat = NaN.
        correlations[:, :, :, i] = np.where(
            (cb_denominator > 0) & (cb == 0),
            0,
            correlations[:, :, :, i]
        )

    return correlations, lags

def plot_correlation_window(row, comparison_data, metric, rpm=False):
    comp = comparison_data[row["comparison_id"]]
    idx = int(row["window_idx"])

    ip_A = comp["ip_A"][idx].copy()
    ip_B = comp["ip_B"][idx].copy()
    values = comp[metric][idx]

    if rpm:
        ip_A *= 1e6 / comp["ip_A_library_sizes"][:, None]
        ip_B *= 1e6 / comp["ip_B_library_sizes"][:, None]

    plt.figure(figsize=(10, 4))

    for i, signal in enumerate(ip_A):
        plt.plot(signal, label=f"{row['experiment_A']} rep {i+1}")

    for i, signal in enumerate(ip_B):
        plt.plot(signal, linestyle="--", label=f"{row['experiment_B']} rep {i+1}")

    plt.title(
        f"{row['chrom']}:{row['start']}-{row['end']} ({row['strand']}) | "
        f"{metric} mean={row[f'{metric}_mean']:.3f}, SD={row[f'{metric}_sd']:.3f}\n"
        f"Replicate values: {np.round(values.flatten(), 3)}"
    )

    plt.xlabel("Position in window")
    plt.ylabel("RPM" if rpm else "Raw IP signal")
    plt.legend()
    plt.show()

def plot_xcorr_window(row, comparison_data, metric="c17"):
    comp = comparison_data[row["comparison_id"]]
    idx = int(row["window_idx"])

    signal_A = comp["signal_A"][idx]
    signal_B = comp["signal_B"][idx]
    xcorr = comp[metric][idx]
    lags = comp[f"{metric}_lags"]

    positions = np.arange(signal_A.shape[1]) - signal_A.shape[1] // 2

    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(xcorr, axis=(0, 1))

    best_lag = row[f"{metric}_best_lag"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Original transformed tracks.
    for i, signal in enumerate(signal_A):
        axes[0].plot(positions, signal, label=f"{row['experiment_A']} rep {i+1}")
    for i, signal in enumerate(signal_B):
        axes[0].plot(positions, signal, linestyle="--", label=f"{row['experiment_B']} rep {i+1}")

    axes[0].axvline(0, linestyle="--")
    axes[0].set_xlim(-150, 150)
    axes[0].set_ylabel("Abundance-weighted enrichment")
    axes[0].set_title("Original tracks")
    axes[0].legend()

    # C17 cross-correlation.
    for a in range(xcorr.shape[0]):
        for b in range(xcorr.shape[1]):
            axes[1].plot(lags, xcorr[a, b], alpha=0.3)

    axes[1].plot(lags, mean_curve, linewidth=2, label="Mean")
    axes[1].axvline(0, linestyle="--")
    axes[1].axvline(best_lag, linestyle=":")
    axes[1].set_xlim(-150, 150)
    axes[1].set_xlabel("Lag (nt)")
    axes[1].set_ylabel("C17")
    axes[1].set_title("Cross-correlation")
    axes[1].legend()

    # Tracks shifted to optimal alignment.
    for i, signal in enumerate(signal_A):
        axes[2].plot(positions, signal, label=f"{row['experiment_A']} rep {i+1}")

    shifted_positions = positions - best_lag
    for i, signal in enumerate(signal_B):
        axes[2].plot(shifted_positions, signal, linestyle="--", label=f"{row['experiment_B']} rep {i+1}")

    axes[2].axvline(0, linestyle="--")
    axes[2].set_xlim(-150, 150)
    axes[2].set_xlabel("Position relative to window center (nt)")
    axes[2].set_ylabel("Abundance-weighted enrichment")
    axes[2].set_title(f"Tracks after shifting B by {-best_lag:+.0f} nt")
    axes[2].legend()

    fig.suptitle(
        f"{row['chrom']}:{row['start']}-{row['end']} ({row['strand']}) | "
        f"max={row[f'{metric}_max']:.3f}, "
        f"best lag={best_lag:.0f}, "
        f"zero={row[f'{metric}_zero']:.3f}, "
        f"gain={row[f'{metric}_gain']:.3f}"
    )

    plt.tight_layout()
    plt.show()

# def plot_xcorr_window(row, comparison_data, metric="pearson_xcorr", rpm=False):
#     comp = comparison_data[row["comparison_id"]]
#     idx = int(row["window_idx"])

#     ip_A = comp["ip_A"][idx].copy()
#     ip_B = comp["ip_B"][idx].copy()
#     xcorr = comp[metric][idx]
#     lags = comp[f"{metric}_lags"]

#     if rpm:
#         ip_A *= 1e6 / comp["ip_A_library_sizes"][:, None]
#         ip_B *= 1e6 / comp["ip_B_library_sizes"][:, None]

#     positions = np.arange(ip_A.shape[1]) - ip_A.shape[1] // 2

#     with np.errstate(invalid="ignore"):
#         mean_curve = np.nanmean(xcorr, axis=(0, 1))

#     best_lag = row[f"{metric}_best_lag"]

#     fig, axes = plt.subplots(3, 1, figsize=(10, 10))

#     # Original tracks.
#     for i, signal in enumerate(ip_A):
#         axes[0].plot(positions, signal, label=f"{row['experiment_A']} rep {i+1}")
#     for i, signal in enumerate(ip_B):
#         axes[0].plot(positions, signal, linestyle="--", label=f"{row['experiment_B']} rep {i+1}")

#     axes[0].axvline(0, linestyle="--")
#     axes[0].set_xlim(-150, 150)
#     axes[0].set_ylabel("RPM" if rpm else "Raw IP signal")
#     axes[0].set_title("Original tracks")
#     axes[0].legend()

#     # Cross-correlation.
#     for a in range(xcorr.shape[0]):
#         for b in range(xcorr.shape[1]):
#             axes[1].plot(lags, xcorr[a, b], alpha=0.3)

#     axes[1].plot(lags, mean_curve, linewidth=2, label="Mean")
#     axes[1].axvline(0, linestyle="--")
#     axes[1].axvline(best_lag, linestyle=":")
#     axes[1].set_xlim(-150, 150)
#     axes[1].set_xlabel("Lag (nt)")
#     axes[1].set_ylabel("CCC" if metric.startswith("ccc") else "Pearson correlation")
#     axes[1].set_title("Cross-correlation")
#     axes[1].legend()

#     # Tracks shifted to optimal alignment.
#     for i, signal in enumerate(ip_A):
#         axes[2].plot(positions, signal, label=f"{row['experiment_A']} rep {i+1}")

#     shifted_positions = positions - best_lag
#     for i, signal in enumerate(ip_B):
#         axes[2].plot(shifted_positions, signal, linestyle="--", label=f"{row['experiment_B']} rep {i+1}")

#     axes[2].axvline(0, linestyle="--")
#     axes[2].set_xlim(-150, 150)
#     axes[2].set_xlabel("Position relative to window center (nt)")
#     axes[2].set_ylabel("RPM" if rpm else "Raw IP signal")
#     axes[2].set_title(f"Tracks after shifting B by {-best_lag:+.0f} nt")
#     axes[2].legend()

#     fig.suptitle(
#         f"{row['chrom']}:{row['start']}-{row['end']} ({row['strand']}) | "
#         f"max={row[f'{metric}_max']:.3f}, "
#         f"best lag={best_lag:.0f}, "
#         f"zero={row[f'{metric}_zero']:.3f}, "
#         f"gain={row[f'{metric}_gain']:.3f}"
#     )

#     plt.tight_layout()
#     plt.show()

################### Old correlation functions (retained for comparison) ###############################

def calculate_pearson(ip_A, ip_B):
    n_windows, n_rep_A, _ = ip_A.shape
    n_rep_B = ip_B.shape[1]

    correlations = np.full((n_windows, n_rep_A, n_rep_B), np.nan, dtype=np.float32)

    for w in range(n_windows):
        for a in range(n_rep_A):
            for b in range(n_rep_B):
                signal_A = ip_A[w, a]
                signal_B = ip_B[w, b]

                if np.std(signal_A) == 0 or np.std(signal_B) == 0:
                    continue

                correlations[w, a, b] = pearsonr(signal_A, signal_B).statistic

    return correlations

def calculate_cosine(ip_A, ip_B):
    n_windows, n_rep_A, _ = ip_A.shape
    n_rep_B = ip_B.shape[1]

    similarities = np.full((n_windows, n_rep_A, n_rep_B), np.nan, dtype=np.float32)

    for w in range(n_windows):
        for a in range(n_rep_A):
            for b in range(n_rep_B):
                signal_A = ip_A[w, a]
                signal_B = ip_B[w, b]

                if np.linalg.norm(signal_A) == 0 or np.linalg.norm(signal_B) == 0:
                    continue

                similarities[w, a, b] = 1 - cosine(signal_A, signal_B)

    return similarities

def calculate_kernel_correlation(ip_A, ip_B, sigma=10):
    n_windows, n_rep_A, _ = ip_A.shape
    n_rep_B = ip_B.shape[1]

    correlations = np.full((n_windows, n_rep_A, n_rep_B), np.nan, dtype=np.float32)

    for w in range(n_windows):
        for a in range(n_rep_A):
            signal_A = ip_A[w, a]
            centered_A = signal_A - signal_A.mean()

            if np.std(signal_A) == 0:
                continue

            smooth_A = gaussian_filter1d(centered_A, sigma=sigma, mode="constant")
            self_A = np.sum(centered_A * smooth_A)

            for b in range(n_rep_B):
                signal_B = ip_B[w, b]

                if np.std(signal_B) == 0:
                    continue

                centered_B = signal_B - signal_B.mean()
                smooth_B = gaussian_filter1d(centered_B, sigma=sigma, mode="constant")

                cross = np.sum(centered_A * smooth_B)
                self_B = np.sum(centered_B * smooth_B)

                denominator = np.sqrt(self_A * self_B)

                if denominator > 0:
                    correlations[w, a, b] = cross / denominator

    return correlations

def calculate_pearson_cross_correlation(ip_A, ip_B, max_lag=30):
    n_windows, n_rep_A, _ = ip_A.shape
    n_rep_B = ip_B.shape[1]
    lags = np.arange(-max_lag, max_lag + 1)

    correlations = np.full(
        (n_windows, n_rep_A, n_rep_B, len(lags)),
        np.nan,
        dtype=np.float32
    )

    for i, lag in enumerate(lags):
        if lag > 0:
            A = ip_A[:, :, :-lag]
            B = ip_B[:, :, lag:]
        elif lag < 0:
            A = ip_A[:, :, -lag:]
            B = ip_B[:, :, :lag]
        else:
            A = ip_A
            B = ip_B

        A_centered = A - A.mean(axis=2, keepdims=True)
        B_centered = B - B.mean(axis=2, keepdims=True)

        numerator = np.einsum("wap,wbp->wab", A_centered, B_centered)

        sumsq_A = np.sum(A_centered ** 2, axis=2)
        sumsq_B = np.sum(B_centered ** 2, axis=2)

        denominator = np.sqrt(
            sumsq_A[:, :, None] * sumsq_B[:, None, :]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            correlations[:, :, :, i] = np.where(
                denominator > 0,
                numerator / denominator,
                np.nan
            )

    return correlations, lags
    
def calculate_kernel_correlation(ip_A, ip_B, sigma=10):
    n_windows, n_rep_A, _ = ip_A.shape
    n_rep_B = ip_B.shape[1]

    correlations = np.full((n_windows, n_rep_A, n_rep_B), np.nan, dtype=np.float32)

    for w in range(n_windows):
        for a in range(n_rep_A):
            signal_A = ip_A[w, a]
            centered_A = signal_A - signal_A.mean()

            if np.std(signal_A) == 0:
                continue

            smooth_A = gaussian_filter1d(centered_A, sigma=sigma, mode="constant")
            self_A = np.sum(centered_A * smooth_A)

            for b in range(n_rep_B):
                signal_B = ip_B[w, b]

                if np.std(signal_B) == 0:
                    continue

                centered_B = signal_B - signal_B.mean()
                smooth_B = gaussian_filter1d(centered_B, sigma=sigma, mode="constant")

                cross = np.sum(centered_A * smooth_B)
                self_B = np.sum(centered_B * smooth_B)

                denominator = np.sqrt(self_A * self_B)

                if denominator > 0:
                    correlations[w, a, b] = cross / denominator

    return correlations

def calculate_pearson_cross_correlation(ip_A, ip_B, max_lag=30):
    n_windows, n_rep_A, _ = ip_A.shape
    n_rep_B = ip_B.shape[1]
    lags = np.arange(-max_lag, max_lag + 1)

    correlations = np.full(
        (n_windows, n_rep_A, n_rep_B, len(lags)),
        np.nan,
        dtype=np.float32
    )

    for i, lag in enumerate(lags):
        if lag > 0:
            A = ip_A[:, :, :-lag]
            B = ip_B[:, :, lag:]
        elif lag < 0:
            A = ip_A[:, :, -lag:]
            B = ip_B[:, :, :lag]
        else:
            A = ip_A
            B = ip_B

        A_centered = A - A.mean(axis=2, keepdims=True)
        B_centered = B - B.mean(axis=2, keepdims=True)

        numerator = np.einsum("wap,wbp->wab", A_centered, B_centered)

        sumsq_A = np.sum(A_centered ** 2, axis=2)
        sumsq_B = np.sum(B_centered ** 2, axis=2)

        denominator = np.sqrt(
            sumsq_A[:, :, None] * sumsq_B[:, None, :]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            correlations[:, :, :, i] = np.where(
                denominator > 0,
                numerator / denominator,
                np.nan
            )

    return correlations, lags

def calculate_ccc_cross_correlation(ip_A, ip_B, library_sizes_A, library_sizes_B, max_lag=30):
    lags = np.arange(-max_lag, max_lag + 1)

    rpm_A = ip_A * (1e6 / library_sizes_A[None, :, None])
    rpm_B = ip_B * (1e6 / library_sizes_B[None, :, None])

    correlations = np.full(
        (len(ip_A), ip_A.shape[1], ip_B.shape[1], len(lags)),
        np.nan,
        dtype=np.float32
    )

    for i, lag in enumerate(lags):
        if lag > 0:
            A = rpm_A[:, :, :-lag]
            B = rpm_B[:, :, lag:]
        elif lag < 0:
            A = rpm_A[:, :, -lag:]
            B = rpm_B[:, :, :lag]
        else:
            A = rpm_A
            B = rpm_B

        mean_A = A.mean(axis=2)
        mean_B = B.mean(axis=2)

        A_centered = A - mean_A[:, :, None]
        B_centered = B - mean_B[:, :, None]

        var_A = np.mean(A_centered ** 2, axis=2)
        var_B = np.mean(B_centered ** 2, axis=2)
        covariance = np.einsum("wap,wbp->wab", A_centered, B_centered) / A.shape[2]

        denominator = (
            var_A[:, :, None]
            + var_B[:, None, :]
            + (mean_A[:, :, None] - mean_B[:, None, :]) ** 2
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            correlations[:, :, :, i] = np.where(
                denominator > 0,
                2 * covariance / denominator,
                np.nan
            )

    return correlations, lags

# def plot_xcorr_window(row, comparison_data, metric="pearson_xcorr"):
#     comp = comparison_data[row["comparison_id"]]
#     idx = int(row["window_idx"])

#     ip_A = comp["ip_A"][idx]
#     ip_B = comp["ip_B"][idx]
#     xcorr = comp[metric][idx]
#     lags = comp[f"{metric}_lags"]

#     with np.errstate(invalid="ignore"):
#         mean_curve = np.nanmean(xcorr, axis=(0, 1))

#     fig, axes = plt.subplots(2, 1, figsize=(10, 7))

#     for i, signal in enumerate(ip_A):
#         axes[0].plot(signal, label=f"{row['experiment_A']} rep {i+1}")

#     for i, signal in enumerate(ip_B):
#         axes[0].plot(signal, linestyle="--", label=f"{row['experiment_B']} rep {i+1}")

#     axes[0].set_ylabel("Raw IP signal")
#     axes[0].legend()

#     for a in range(xcorr.shape[0]):
#         for b in range(xcorr.shape[1]):
#             axes[1].plot(lags, xcorr[a, b], alpha=0.3)

#     axes[1].plot(lags, mean_curve, linewidth=2, label="Mean")
#     axes[1].axvline(0, linestyle="--")
#     axes[1].axvline(row["pearson_xcorr_best_lag"], linestyle=":")
#     axes[1].set_xlabel("Lag (nt)")
#     axes[1].set_ylabel("Pearson correlation")
#     axes[1].legend()

#     fig.suptitle(
#         f"{row['chrom']}:{row['start']}-{row['end']} ({row['strand']}) | "
#         f"max={row['pearson_xcorr_max']:.3f}, "
#         f"best lag={row['pearson_xcorr_best_lag']:.0f}, "
#         f"zero={row['pearson_xcorr_zero']:.3f}, "
#         f"gain={row['pearson_xcorr_gain']:.3f}"
#     )

#     plt.tight_layout()
#     plt.show()