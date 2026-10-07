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

def correspondence_matrix(signal_A, signal_B, core_size=300):
    """
    Build a position-by-position correspondence matrix for each window.

    Each cell M[i, j] contains the shared positive signal between
    position i in A and position j in B, currently defined as min(A_i, B_j).

    Replicates are averaged before constructing the matrix.
    """

    if signal_A.shape[0] != signal_B.shape[0]:
        raise ValueError("A and B must contain the same number of windows.")

    if signal_A.shape[2] != signal_B.shape[2]:
        raise ValueError("A and B must have the same positional length.")

    context = (signal_A.shape[2] - core_size) // 2
    core_start = context
    core_end = context + core_size

    # Central analytical 300-nt window.
    A = signal_A[:, :, core_start:core_end].mean(axis=1)
    B = signal_B[:, :, core_start:core_end].mean(axis=1)

    # Compare every position in A with every position in B.
    matrices = np.minimum(
        A[:, :, None],
        B[:, None, :]
    ).astype(np.float32)

    return matrices

def plot_correspondence_matrix(matrix, title=None):
    plt.figure(figsize=(7, 6))

    plt.imshow(
        matrix,
        origin="lower",
        aspect="equal"
    )

    plt.colorbar(label="Shared signal")
    plt.xlabel("B position")
    plt.ylabel("A position")

    # Zero-lag diagonal.
    plt.plot(
        [0, matrix.shape[1] - 1],
        [0, matrix.shape[0] - 1],
        linestyle="--",
        linewidth=1
    )

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def plot_category(category):
    subset = metadata[
        metadata["zoo_category"] == category
    ].reset_index(drop=True)

    if len(subset) == 0:
        print(f"No windows found for category: {category}")
        return

    for _, row in subset.iterrows():
        matrix = comparison_data[
            row["comparison_id"]
        ]["correspondence_matrix"][row["window_idx"]]

        title = (
            f'{row["experiment_A"]} vs {row["experiment_B"]}\n'
            f'{row["chrom"]}:{row["start"]}-{row["end"]}'
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        im = plot_correspondence_matrix(
            matrix,
            title=title,
            ax=ax
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Shared signal"
        )

        plt.tight_layout()
        plt.show()

def rigid_correspondences(
    matrix,
    threshold,
    min_length=5,
    min_score=0,
    max_paths=None
):
    """
    Find multiple non-overlapping rigid local correspondences.

    Paths are constrained to:
        (i, j) -> (i + 1, j + 1)

    Candidate segments are found independently along every diagonal,
    ranked by score, then greedily selected so that claimed A/B regions
    are not reused.
    """

    score_matrix = matrix - threshold

    n_A, n_B = matrix.shape
    candidates = []

    # Search every diagonal.
    for offset in range(-(n_A - 1), n_B):

        diagonal = np.diagonal(score_matrix, offset=offset)

        current_score = 0.0
        current_start = 0

        best_run_score = 0.0
        best_run_start = None
        best_run_end = None

        for k, value in enumerate(diagonal):
            current_score += value

            # Track best endpoint within this positive run.
            if current_score > best_run_score:
                best_run_score = current_score
                best_run_start = current_start
                best_run_end = k

            # End of local correspondence block.
            if current_score <= 0:
                if best_run_start is not None:
                    if offset >= 0:
                        start_A = best_run_start
                        start_B = best_run_start + offset
                        end_A = best_run_end
                        end_B = best_run_end + offset
                    else:
                        start_A = best_run_start - offset
                        start_B = best_run_start
                        end_A = best_run_end - offset
                        end_B = best_run_end

                    length = end_A - start_A + 1

                    if length >= min_length and best_run_score >= min_score:
                        candidates.append({
                            "start_A": start_A,
                            "start_B": start_B,
                            "end_A": end_A,
                            "end_B": end_B,
                            "offset": offset,
                            "length": length,
                            "score": best_run_score
                        })

                current_score = 0.0
                current_start = k + 1

                best_run_score = 0.0
                best_run_start = None
                best_run_end = None

        # Catch a run reaching the end of the diagonal.
        if best_run_start is not None:
            if offset >= 0:
                start_A = best_run_start
                start_B = best_run_start + offset
                end_A = best_run_end
                end_B = best_run_end + offset
            else:
                start_A = best_run_start - offset
                start_B = best_run_start
                end_A = best_run_end - offset
                end_B = best_run_end

            length = end_A - start_A + 1

            if length >= min_length and best_run_score >= min_score:
                candidates.append({
                    "start_A": start_A,
                    "start_B": start_B,
                    "end_A": end_A,
                    "end_B": end_B,
                    "offset": offset,
                    "length": length,
                    "score": best_run_score
                })

    # Strongest candidates first.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Greedily keep candidates that do not reuse A or B regions.
    selected = []

    used_A = np.zeros(n_A, dtype=bool)
    used_B = np.zeros(n_B, dtype=bool)

    for candidate in candidates:

        A_slice = slice(
            candidate["start_A"],
            candidate["end_A"] + 1
        )

        B_slice = slice(
            candidate["start_B"],
            candidate["end_B"] + 1
        )

        if used_A[A_slice].any() or used_B[B_slice].any():
            continue

        selected.append(candidate)

        used_A[A_slice] = True
        used_B[B_slice] = True

        if max_paths is not None and len(selected) >= max_paths:
            break

    # Add mean raw matrix support.
    for path in selected:
        values = matrix[
            np.arange(path["start_A"], path["end_A"] + 1),
            np.arange(path["start_B"], path["end_B"] + 1)
        ]

        path["mean_support"] = values.mean()

    return selected