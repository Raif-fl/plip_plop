from scipy.stats import pearsonr
import os
import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy import ndimage

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

def displacement_preference(displacement, scale=10):
    """
    Prefer near-zero displacement strongly, while making differences
    between already-large displacements matter progressively less.
    """
    return 1 / (1 + (abs(displacement) / scale) ** 2)

def island_central_ridge(matrix, region, ridge_fraction=0.9, ridge_smoothing=4.0):
    """Extract a noise-tolerant crest using geometry, then morphology for square ties."""
    coords = np.argwhere(region)
    A_min, A_max = coords[:, 0].min(), coords[:, 0].max()
    B_min, B_max = coords[:, 1].min(), coords[:, 1].max()
    A_width, B_width = A_max - A_min + 1, B_max - B_min + 1

    def ridge_run(profile):
        smooth = gaussian_filter1d(profile, ridge_smoothing, mode="nearest") if ridge_smoothing > 0 else profile
        threshold = ridge_fraction * smooth.max()
        start = end = int(np.argmax(smooth))
        while start > 0 and smooth[start - 1] >= threshold:
            start -= 1
        while end < len(smooth) - 1 and smooth[end + 1] >= threshold:
            end += 1
        return threshold, start, end

    A_positions = np.arange(A_min, A_max + 1)
    B_positions = np.arange(B_min, B_max + 1)
    A_profile = np.array([matrix[A, region[A]].max() for A in A_positions])
    B_profile = np.array([matrix[region[:, B], B].max() for B in B_positions])
    A_threshold, A_start, A_end = ridge_run(A_profile)
    B_threshold, B_start, B_end = ridge_run(B_profile)

    # Geometry decides non-square islands; signal morphology breaks square ties.
    if A_width > B_width or (A_width == B_width and A_end - A_start > B_end - B_start):
        orientation = "vertical"
        long_positions, ridge_start, ridge_end = A_positions, A_start, A_end
        ridge_threshold = A_threshold
    else:
        orientation = "horizontal"
        long_positions, ridge_start, ridge_end = B_positions, B_start, B_end
        ridge_threshold = B_threshold

    ridge_long = long_positions[ridge_start:ridge_end + 1].astype(float)
    ridge_region = region.copy()
    if orientation == "vertical":
        ridge_region[:long_positions[ridge_start]] = False
        ridge_region[long_positions[ridge_end] + 1:] = False
        short_profile = np.where(ridge_region, matrix, -np.inf).max(axis=0)
        valid = np.isfinite(short_profile)
        short_peak = short_profile[valid].max()
        short_positions = np.flatnonzero(valid & (short_profile >= ridge_fraction * short_peak))
        ridge_short = np.average(short_positions, weights=short_profile[short_positions])
        ridge_A = ridge_long
        ridge_B = np.full(len(ridge_long), ridge_short)
    else:
        ridge_region[:, :long_positions[ridge_start]] = False
        ridge_region[:, long_positions[ridge_end] + 1:] = False
        short_profile = np.where(ridge_region, matrix, -np.inf).max(axis=1)
        valid = np.isfinite(short_profile)
        short_peak = short_profile[valid].max()
        short_positions = np.flatnonzero(valid & (short_profile >= ridge_fraction * short_peak))
        ridge_short = np.average(short_positions, weights=short_profile[short_positions])
        ridge_B = ridge_long
        ridge_A = np.full(len(ridge_long), ridge_short)

    return {
        "ridge_A": ridge_A,
        "ridge_B": ridge_B,
        "ridge_orientation": orientation,
        "ridge_area": len(ridge_A),
        "ridge_threshold": ridge_threshold,
    }


def island_zero_lag_distance(island):
    """Minimum distance between the extracted central ridge and A = B."""
    return float(np.min(np.abs(np.asarray(island["ridge_A"]) - np.asarray(island["ridge_B"]))))

def island_axis_profile(matrix, region, axis, smoothing):
    """Return an island-local maximum-support profile along A or B."""
    if axis == "A":
        positions = np.flatnonzero(region.any(axis=1))
        profile = np.array([matrix[position, region[position]].max() for position in positions])
    elif axis == "B":
        positions = np.flatnonzero(region.any(axis=0))
        profile = np.array([matrix[region[:, position], position].max() for position in positions])
    else:
        raise ValueError("axis must be 'A' or 'B'.")

    smooth = gaussian_filter1d(profile, smoothing, mode="nearest") if smoothing > 0 else profile
    return positions, smooth

def best_island_split(matrix, region, seed_mask, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5):
    """Find the strongest valid island-local valley along either axis."""
    candidates = []

    for axis in ("A", "B"):
        positions, profile = island_axis_profile(matrix, region, axis, split_smoothing)
        if len(profile) < 2 * split_min_peak_distance + 1:
            continue

        peaks, _ = find_peaks(profile, distance=split_min_peak_distance)
        if len(peaks) < 2:
            continue

        peaks = peaks[profile[peaks] >= split_min_peak_fraction * profile.max()]
        for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
            if positions[right_peak] - positions[left_peak] < split_min_peak_distance:
                continue

            valley_segment = profile[left_peak:right_peak + 1]
            valley = left_peak + int(np.argmin(valley_segment))
            smaller_peak = min(profile[left_peak], profile[right_peak])
            valley_depth = 1 - profile[valley] / smaller_peak if smaller_peak > 0 else 0.0
            if valley_depth < split_prominence_fraction:
                continue

            cut = int(positions[valley])
            coordinates = np.indices(region.shape)[0 if axis == "A" else 1]
            child_1 = region & (coordinates <= cut)
            child_2 = region & (coordinates > cut)
            seed_area_1 = int(np.count_nonzero(seed_mask & child_1))
            seed_area_2 = int(np.count_nonzero(seed_mask & child_2))
            if seed_area_1 < split_min_seed_area or seed_area_2 < split_min_seed_area:
                continue

            candidates.append({
                "axis": axis, "cut": cut, "valley_depth": float(valley_depth),
                "smaller_peak": float(smaller_peak), "children": (child_1, child_2),
            })

    return max(candidates, key=lambda x: (x["valley_depth"], x["smaller_peak"])) if candidates else None

def split_island_region(matrix, region, seed_mask, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5):
    """Recursively split an overmerged island at its strongest valid axis valley."""
    pending = [region]
    finished = []

    while pending:
        current = pending.pop()
        split = best_island_split(
            matrix, current, seed_mask,
            split_prominence_fraction=split_prominence_fraction,
            split_min_peak_fraction=split_min_peak_fraction,
            split_min_peak_distance=split_min_peak_distance,
            split_smoothing=split_smoothing,
            split_min_seed_area=split_min_seed_area)
        if split is None:
            finished.append(current)
        else:
            pending.extend(split["children"])

    return finished


def correspondence_islands(matrix, seed_threshold_fraction=0.4, growth_threshold_fraction=0.39, max_weak_distance=2, gap_tolerance=1, shape_balance_power=1.05, ridge_fraction=0.9, ridge_smoothing=4.0, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5):
    """
    Identify correspondence-support islands using:

    1. A high threshold to define strong seed regions.
    2. A lower threshold to define weaker surrounding support.
    3. Morphological closing to tolerate very short dips below the
       growth threshold.
    4. A maximum distance from strong seed support to prevent long
       weak plateaus from growing indefinitely.
    5. Island-local valley splitting to separate overmerged peaks.
    6. A high-support crest to define the central ridge used for zero lag.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional.")

    if growth_threshold_fraction > seed_threshold_fraction:
        raise ValueError(
            "growth_threshold_fraction must be <= seed_threshold_fraction."
        )

    if max_weak_distance < 0:
        raise ValueError("max_weak_distance must be >= 0.")

    if gap_tolerance < 0:
        raise ValueError("gap_tolerance must be >= 0.")

    if shape_balance_power < 0:
        raise ValueError("shape_balance_power must be >= 0.")

    if not 0 < ridge_fraction <= 1:
        raise ValueError("ridge_fraction must be > 0 and <= 1.")

    if ridge_smoothing < 0:
        raise ValueError("ridge_smoothing must be >= 0.")

    if not 0 <= split_prominence_fraction <= 1:
        raise ValueError("split_prominence_fraction must be between 0 and 1.")

    if not 0 < split_min_peak_fraction <= 1:
        raise ValueError("split_min_peak_fraction must be > 0 and <= 1.")

    if split_min_peak_distance < 1:
        raise ValueError("split_min_peak_distance must be >= 1.")

    if split_smoothing < 0:
        raise ValueError("split_smoothing must be >= 0.")

    if split_min_seed_area < 1:
        raise ValueError("split_min_seed_area must be >= 1.")

    max_value = matrix.max()

    if max_value <= 0:
        empty_bool = np.zeros_like(matrix, dtype=bool)
        empty_labels = np.zeros_like(matrix, dtype=np.int32)

        return empty_bool, empty_bool, empty_labels, []

    seed_threshold = seed_threshold_fraction * max_value
    growth_threshold = growth_threshold_fraction * max_value

    # Strong support required to initiate an island.
    seed_mask = matrix > seed_threshold

    # Weaker support that may be incorporated into an island.
    growth_mask = matrix > growth_threshold

    # Tolerate very short interruptions below the growth threshold.
    if gap_tolerance > 0:
        structure = np.ones((3, 3), dtype=bool)

        growth_mask = ndimage.binary_closing(
            growth_mask,
            structure=structure,
            iterations=gap_tolerance
        )

    # Distance from every cell to the nearest strong seed cell.
    distance_from_seed = ndimage.distance_transform_edt(~seed_mask)

    # Keep only growth-support cells close enough to a strong seed.
    island_mask = (
        growth_mask &
        (distance_from_seed <= max_weak_distance)
    )

    # Seed cells are always retained.
    island_mask |= seed_mask

    # Fill enclosed holes inside otherwise continuous islands.
    island_mask = ndimage.binary_fill_holes(island_mask)

    # 8-connectivity.
    structure = np.ones((3, 3), dtype=bool)

    raw_labels, n_regions = ndimage.label(
        island_mask,
        structure=structure
    )

    split_regions = []
    for raw_id in range(1, n_regions + 1):
        region = raw_labels == raw_id

        # Only retain regions that contain actual strong seed support.
        if not np.any(seed_mask & region):
            continue

        split_regions.extend(split_island_region(
            matrix, region, seed_mask,
            split_prominence_fraction=split_prominence_fraction,
            split_min_peak_fraction=split_min_peak_fraction,
            split_min_peak_distance=split_min_peak_distance,
            split_smoothing=split_smoothing,
            split_min_seed_area=split_min_seed_area))

    labels = np.zeros_like(raw_labels, dtype=np.int32)
    islands = []

    for island_id, region in enumerate(split_regions, start=1):

        labels[region] = island_id

        coords = np.argwhere(region)

        A_positions = coords[:, 0]
        B_positions = coords[:, 1]

        values = matrix[
            A_positions,
            B_positions
        ]

        seed_coords = np.argwhere(
            seed_mask & region
        )

        # Describe island geometry.
        A_min = A_positions.min()
        A_max = A_positions.max()

        B_min = B_positions.min()
        B_max = B_positions.max()

        A_width = A_max - A_min + 1
        B_width = B_max - B_min + 1

        A_center = (A_min + A_max) / 2
        B_center = (B_min + B_max) / 2

        displacement = B_center - A_center
        total_support = values.sum()
        shape_balance = min(A_width, B_width) / max(A_width, B_width)
        selection_score = total_support * shape_balance ** shape_balance_power

        island = {
            "island_id": island_id,

            "A_min": A_min,
            "A_max": A_max,
            "B_min": B_min,
            "B_max": B_max,

            "A_width": A_width,
            "B_width": B_width,

            "A_center": A_center,
            "B_center": B_center,

            "displacement": displacement,
            "abs_displacement": abs(displacement),

            "area": len(coords),
            "max_support": values.max(),
            "mean_support": values.mean(),
            "total_support": total_support,
            "shape_balance": shape_balance,
            "selection_score": selection_score,
            "seed_area": len(seed_coords),
        }

        # Extract the high-support crest and measure that crest against zero lag.
        island.update(island_central_ridge(matrix, region, ridge_fraction=ridge_fraction, ridge_smoothing=ridge_smoothing))
        island["zero_lag_distance"] = island_zero_lag_distance(island)
        island["signed_lag"] = float(np.median(np.asarray(island["ridge_B"]) - np.asarray(island["ridge_A"])))
        island["ridge_A_min"] = float(np.min(island["ridge_A"]))
        island["ridge_A_max"] = float(np.max(island["ridge_A"]))
        island["ridge_B_min"] = float(np.min(island["ridge_B"]))
        island["ridge_B_max"] = float(np.max(island["ridge_B"]))

        islands.append(island)

    return seed_mask, island_mask, labels, islands

def island_conflicts(labels, islands):
    """
    Identify competing correspondence islands.

    Two islands conflict if they occupy any of the same A positions
    or any of the same B positions.

    Returns
    -------
    conflicts : list of dict
        One record per conflicting island pair.
    """

    projections = {}

    for island in islands:
        island_id = island["island_id"]
        region = labels == island_id

        projections[island_id] = {
            # Which A positions are claimed by this island?
            "A": region.any(axis=1),

            # Which B positions are claimed by this island?
            "B": region.any(axis=0),
        }

    conflicts = []

    for i in range(len(islands)):
        id_i = islands[i]["island_id"]

        for j in range(i + 1, len(islands)):
            id_j = islands[j]["island_id"]

            A_overlap = np.any(
                projections[id_i]["A"] &
                projections[id_j]["A"]
            )

            B_overlap = np.any(
                projections[id_i]["B"] &
                projections[id_j]["B"]
            )

            if not (A_overlap or B_overlap):
                continue

            if A_overlap and B_overlap:
                conflict_type = "A_and_B"
            elif A_overlap:
                conflict_type = "A"
            else:
                conflict_type = "B"

            conflicts.append({
                "island_1": id_i,
                "island_2": id_j,
                "A_overlap": A_overlap,
                "B_overlap": B_overlap,
                "conflict_type": conflict_type,
            })

    return conflicts

def ridge_conflicts(island_1, island_2):
    """Return whether two ridges claim any of the same A or B positions."""
    A_overlap = max(island_1["ridge_A_min"], island_2["ridge_A_min"]) <= min(island_1["ridge_A_max"], island_2["ridge_A_max"])
    B_overlap = max(island_1["ridge_B_min"], island_2["ridge_B_min"]) <= min(island_1["ridge_B_max"], island_2["ridge_B_max"])
    return A_overlap or B_overlap

def strongest_ridge_compatible_subset(islands):
    """Find the strongest ordered subset with non-overlapping ridge projections.

    A coherent displacement preserves position order in A and B, so this is a
    weighted longest-chain calculation rather than exponential subset search.
    """
    if not islands:
        return (), 0.0

    ordered = sorted(islands, key=lambda x: (x["ridge_A_min"], x["ridge_B_min"], x["island_id"]))
    best_scores = []
    best_paths = []

    for j, island in enumerate(ordered):
        island_score = island.get("selection_score", island["total_support"])
        best_score = island_score
        best_path = (island["island_id"],)

        for i, previous in enumerate(ordered[:j]):
            if previous["ridge_A_max"] >= island["ridge_A_min"] or previous["ridge_B_max"] >= island["ridge_B_min"]:
                continue
            candidate_score = best_scores[i] + island_score
            candidate_path = best_paths[i] + (island["island_id"],)
            if (candidate_score, len(candidate_path)) > (best_score, len(best_path)):
                best_score, best_path = candidate_score, candidate_path

        best_scores.append(best_score)
        best_paths.append(best_path)

    best_index = max(range(len(ordered)), key=lambda i: (best_scores[i], len(best_paths[i])))
    return best_paths[best_index], best_scores[best_index]

def correspondence_modes(islands, mode_lag_tolerance=5, zero_lag_threshold=5):
    """Group islands with similar signed ridge displacement into modes."""
    if mode_lag_tolerance < 0:
        raise ValueError("mode_lag_tolerance must be >= 0.")

    groups = []
    central_islands = [island for island in islands
                       if island["zero_lag_distance"] <= zero_lag_threshold]
    displaced_islands = [island for island in islands
                         if island["zero_lag_distance"] > zero_lag_threshold]

    # Every island touching the zero-lag corridor belongs to one central mode.
    if central_islands:
        weights = [island.get("selection_score", island["total_support"]) for island in central_islands]
        lags = [island["signed_lag"] for island in central_islands]
        central_lag = float(np.average(lags, weights=weights)) if sum(weights) > 0 else float(np.mean(lags))
        groups.append({"islands": central_islands, "signed_lag": central_lag, "zero_lag": True})

    # Displaced modes are clusters of islands with similar signed lag.
    for island in sorted(displaced_islands, key=lambda x: x["signed_lag"]):
        lag = island["signed_lag"]
        matches = [(abs(lag - group["signed_lag"]), i) for i, group in enumerate(groups)
                   if not group["zero_lag"] and abs(lag - group["signed_lag"]) <= mode_lag_tolerance]

        if not matches:
            groups.append({"islands": [island], "signed_lag": lag, "zero_lag": False})
            continue

        _, group_index = min(matches)
        group = groups[group_index]
        group["islands"].append(island)
        weights = [member.get("selection_score", member["total_support"]) for member in group["islands"]]
        lags = [member["signed_lag"] for member in group["islands"]]
        group["signed_lag"] = float(np.average(lags, weights=weights)) if sum(weights) > 0 else float(np.mean(lags))

    modes = []
    for mode_id, group in enumerate(sorted(groups, key=lambda x: x["signed_lag"]), start=1):
        members = group["islands"]
        scores = [island.get("selection_score", island["total_support"]) for island in members]
        compatible_islands, compatible_score = strongest_ridge_compatible_subset(members)
        compatible_ids = set(compatible_islands)
        ridge_conflict_pairs = tuple(
            (members[i]["island_id"], members[j]["island_id"])
            for i in range(len(members)) for j in range(i + 1, len(members))
            if ridge_conflicts(members[i], members[j])
        )
        possible_pairs = len(members) * (len(members) - 1) / 2
        zero_lag_islands = [island["island_id"] for island in members] if group["zero_lag"] else []
        modes.append({
            "mode_id": mode_id,
            "islands": tuple(island["island_id"] for island in members),
            "n_islands": len(members),
            "compatible_islands": compatible_islands,
            "n_independent_islands": len(compatible_islands),
            "ridge_conflicts": ridge_conflict_pairs,
            "conflict_fraction": len(ridge_conflict_pairs) / possible_pairs if possible_pairs else 0.0,
            "signed_lag": group["signed_lag"],
            "lag_min": min(island["signed_lag"] for island in members),
            "lag_max": max(island["signed_lag"] for island in members),
            # Overlapping ridges remain visible but cannot contribute duplicate evidence.
            "selection_score": compatible_score,
            "all_island_score": sum(scores),
            "total_support": sum(island["total_support"] for island in members),
            "compatible_total_support": sum(island["total_support"] for island in members
                                            if island["island_id"] in compatible_ids),
            "zero_lag": group["zero_lag"],
            "zero_lag_islands": tuple(zero_lag_islands),
            "min_zero_lag_distance": min(island["zero_lag_distance"] for island in members),
        })

    return modes

def select_correspondence_modes(islands, conflicts, zero_lag_threshold=5, displacement_fold_threshold=2.0, mode_lag_tolerance=5, mode_score_fraction=0.33):
    """Form modes, resolve central-mode conflicts, and retain viable modes."""
    if not 0 <= mode_score_fraction <= 1:
        raise ValueError("mode_score_fraction must be between 0 and 1.")
    if displacement_fold_threshold <= 0:
        raise ValueError("displacement_fold_threshold must be > 0.")

    # Modes must exist before the central mode is allowed to delete anything.
    provisional_modes = correspondence_modes(
        islands, mode_lag_tolerance=mode_lag_tolerance,
        zero_lag_threshold=zero_lag_threshold)

    if not provisional_modes:
        return {
            "surviving_islands": [], "deleted_island_ids": set(),
            "provisional_modes": [], "candidate_modes": [], "modes": [],
            "central_gate": {"status": "absent"},
            "rejected_mode_ids": set(), "island_modes": {},
        }

    strongest_score = max(mode["selection_score"] for mode in provisional_modes)
    for mode in provisional_modes:
        mode["score_fraction_of_strongest"] = mode["selection_score"] / strongest_score if strongest_score > 0 else 0.0

    central_mode = next((mode for mode in provisional_modes if mode["zero_lag"]), None)
    deleted_island_ids = set()
    central_gate = {"status": "absent"}

    if central_mode is not None:
        central_member_ids = set(central_mode["islands"])
        central_ids = set(central_mode["compatible_islands"])
        conflict_pairs = {frozenset((conflict["island_1"], conflict["island_2"])) for conflict in conflicts}
        conflicting_modes = [
            mode for mode in provisional_modes
            if not mode["zero_lag"] and any(
                frozenset((central_id, displaced_id)) in conflict_pairs
                for central_id in central_ids for displaced_id in mode["compatible_islands"])
        ]
        strongest_conflicting = max(conflicting_modes, key=lambda x: x["selection_score"]) if conflicting_modes else None
        central_is_eligible = central_mode["score_fraction_of_strongest"] >= mode_score_fraction
        central_is_overwhelmed = (
            strongest_conflicting is not None
            and strongest_conflicting["selection_score"] >= displacement_fold_threshold * central_mode["selection_score"]
        )

        if not central_is_eligible:
            deleted_island_ids.update(central_member_ids)
            central_gate = {
                "status": "rejected", "reason": "below_mode_score_fraction",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_compatible_islands": central_mode["compatible_islands"],
                "score_fraction_of_strongest": central_mode["score_fraction_of_strongest"],
            }
        elif central_is_overwhelmed:
            deleted_island_ids.update(central_member_ids)
            central_gate = {
                "status": "rejected", "reason": "overwhelmed_by_displaced_mode",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_compatible_islands": central_mode["compatible_islands"],
                "displaced_mode_id": strongest_conflicting["mode_id"],
                "displaced_score": strongest_conflicting["selection_score"],
                "required_displaced_score": displacement_fold_threshold * central_mode["selection_score"],
            }
        else:
            for conflict in conflicts:
                island_1, island_2 = conflict["island_1"], conflict["island_2"]
                if island_1 in central_ids and island_2 not in central_member_ids:
                    deleted_island_ids.add(island_2)
                elif island_2 in central_ids and island_1 not in central_member_ids:
                    deleted_island_ids.add(island_1)
            central_gate = {
                "status": "retained", "reason": "central_mode_won_conflicts",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_compatible_islands": central_mode["compatible_islands"],
                "strongest_conflicting_mode_id": strongest_conflicting["mode_id"] if strongest_conflicting else None,
                "strongest_conflicting_score": strongest_conflicting["selection_score"] if strongest_conflicting else None,
            }

    surviving_islands = [island for island in islands if island["island_id"] not in deleted_island_ids]
    candidate_modes = correspondence_modes(
        surviving_islands, mode_lag_tolerance=mode_lag_tolerance,
        zero_lag_threshold=zero_lag_threshold)
    strongest_score = max((mode["selection_score"] for mode in candidate_modes), default=0.0)
    for mode in candidate_modes:
        mode["score_fraction_of_strongest"] = mode["selection_score"] / strongest_score if strongest_score > 0 else 0.0

    retained_candidates = [mode for mode in candidate_modes
                           if mode["score_fraction_of_strongest"] >= mode_score_fraction]
    rejected_mode_ids = {mode["mode_id"] for mode in candidate_modes if mode not in retained_candidates}
    modes = []
    for mode_id, candidate in enumerate(retained_candidates, start=1):
        mode = dict(candidate)
        mode["candidate_mode_id"] = candidate["mode_id"]
        mode["mode_id"] = mode_id
        modes.append(mode)
    island_modes = {island_id: mode["mode_id"] for mode in modes for island_id in mode["islands"]}

    return {
        "surviving_islands": surviving_islands,
        "deleted_island_ids": deleted_island_ids,
        "provisional_modes": provisional_modes,
        "candidate_modes": candidate_modes,
        "modes": modes,
        "central_gate": central_gate,
        "rejected_mode_ids": rejected_mode_ids,
        "island_modes": island_modes,
    }

########################### SIMULATION ####################################

def gaussian(x, center, width, height=1.0):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def plateau(x, center, width, edge=5.0, height=1.0):
    left = 1 / (1 + np.exp(-(x - (center - width / 2)) / edge))
    right = 1 / (1 + np.exp((x - (center + width / 2)) / edge))
    return height * left * right


def smooth_noise(rng, scale, n):
    anchor_x = np.linspace(0, n - 1, 32)
    anchors = rng.normal(0, scale, len(anchor_x))
    return np.interp(np.arange(n), anchor_x, anchors)


def draw_count(rng, count_range):
    low, high = count_range
    return int(rng.integers(low, high + 1))


def draw_centers(rng, count, location_range):
    low, high = location_range
    return np.sort(rng.uniform(low, high, count))


def draw_heights(rng, count, signal_level, signal_variation):
    low = signal_level * (1 - signal_variation)
    high = signal_level * (1 + signal_variation)
    return np.maximum(rng.uniform(low, high, count), 0.02)


def draw_widths(rng, count, width_range):
    return rng.uniform(width_range[0], width_range[1], count)


def peak_train(x, centers, widths, heights, n, *, background=0.06):
    values = np.full(n, background, dtype=float)

    for center, width, height in zip(centers, widths, heights):
        values += gaussian(x, center, width, height)

    return values


def primitive(x, name, center, width, height, n):
    background = np.full(n, 0.06)

    if name == "bell":
        return background + gaussian(x, center, width, height)
    if name == "broad":
        return background + gaussian(x, center, width * 2.4, height * 0.82)
    if name == "double":
        return (background
                + gaussian(x, center - width * 0.8, width * 0.65, height * 0.88)
                + gaussian(x, center + width * 0.9, width * 0.8, height * 0.70))
    if name == "plateau":
        return background + plateau(x, center, width * 3.3, max(2, width * 0.25), height)
    if name == "blip":
        return background + gaussian(x, center, max(2, width * 0.22), height)
    if name == "dip":
        return 0.14 + height * 0.75 - gaussian(x, center, width * 0.7, height * 0.82)
    if name == "noise":
        return background

    raise ValueError(f"unknown shape: {name}")


def add_noise(values, rng, noise_level, n):
    fine_noise = rng.normal(0, noise_level, n)
    drift = smooth_noise(rng, noise_level * 0.75, n)
    return values + fine_noise + drift


def shifted_centers(rng, centers, displacement_range, n):
    distances = rng.uniform(displacement_range[0], displacement_range[1], len(centers))
    directions = rng.choice([-1, 1], len(centers))
    return np.clip(centers + distances * directions, 0, n - 1)


def shared_displacement(rng, centers, displacement_range, n):
    min_distance, max_distance = displacement_range
    room_left = np.min(centers)
    room_right = (n - 1) - np.max(centers)
    options = []

    if room_left >= min_distance:
        options.append((-1, min(max_distance, room_left)))
    if room_right >= min_distance:
        options.append((1, min(max_distance, room_right)))

    direction, allowed_max = options[int(rng.integers(len(options)))]
    return direction * rng.uniform(min_distance, allowed_max)


def unrelated_centers(rng, count, existing, location_range, min_distance=25):
    result = []

    for _ in range(count):
        candidates = rng.uniform(location_range[0], location_range[1], 100)
        valid = [c for c in candidates if np.all(np.abs(np.asarray(existing) - c) >= min_distance)]
        result.append(valid[0] if valid else candidates[0])

    return np.sort(result)


def simulate_pair(scenario, rng, signal_level=0.75, signal_variation=0.15, noise_level=0.025, peak_count_range=(1, 4), unrelated_peak_count_range=(1, 3), location_range=(20, 280), width_range=(6, 24), displacement_range=(5, 30), tie_displacement_range=(30, 90), tie_strength_variation=0.03, shape_a="bell", shape_b="bell", n=300):
    x = np.arange(n)

    n_peaks = draw_count(rng, peak_count_range)
    centers = draw_centers(rng, n_peaks, location_range)
    widths = draw_widths(rng, n_peaks, width_range)
    heights = draw_heights(rng, n_peaks, signal_level, signal_variation)
    scenario_noise = noise_level

    if scenario == "easy_agreement":
        # Same peaks with small position, width, and height perturbations.
        a = peak_train(x, centers, widths, heights, n)
        b = peak_train(x, centers + rng.normal(0, 2, n_peaks),
                       widths * rng.uniform(0.90, 1.10, n_peaks),
                       heights * rng.uniform(0.90, 1.10, n_peaks), n)

    elif scenario == "pure_displacement":
        # One displacement is shared by every peak in the window.
        window_displacement = shared_displacement(rng, centers, displacement_range, n)
        a = peak_train(x, centers, widths, heights, n)
        b = peak_train(x, centers + window_displacement, widths, heights, n)

    elif scenario == "width_mismatch":
        # Shared centers with narrow peaks in A and broad peaks in B.
        a = peak_train(x, centers, widths * rng.uniform(0.35, 0.65, n_peaks), heights, n)
        b = peak_train(x, centers + rng.normal(0, 2, n_peaks),
                       widths * rng.uniform(1.8, 3.2, n_peaks), heights, n)

    elif scenario == "local_agreement_plus_unrelated":
        # Shared peaks plus independent extra structure.
        a = peak_train(x, centers, widths, heights, n)
        b = peak_train(x, centers + rng.normal(0, 2, n_peaks), widths, heights, n)

        n_extra_a = draw_count(rng, unrelated_peak_count_range)
        n_extra_b = draw_count(rng, unrelated_peak_count_range)
        centers_a = unrelated_centers(rng, n_extra_a, centers, location_range)
        centers_b = unrelated_centers(rng, n_extra_b, centers, location_range)
        widths_a = draw_widths(rng, n_extra_a, width_range)
        widths_b = draw_widths(rng, n_extra_b, width_range)
        heights_a = draw_heights(rng, n_extra_a, signal_level * 0.60, signal_variation)
        heights_b = draw_heights(rng, n_extra_b, signal_level * 0.60, signal_variation)

        a += peak_train(x, centers_a, widths_a, heights_a, n, background=0)
        b += peak_train(x, centers_b, widths_b, heights_b, n, background=0)

    elif scenario == "broad_signal_plus_peak":
        # Edge-to-edge signal in A with local peaks in B.
        block_height = draw_heights(rng, 1, signal_level, signal_variation)[0]
        a = np.full(n, block_height)
        b = peak_train(x, centers, widths, heights, n)

    elif scenario == "local_anticorrelation":
        # Edge-to-edge signal in B with local depletion beneath A's peaks.
        block_height = draw_heights(rng, 1, signal_level, signal_variation)[0]
        a = peak_train(x, centers, widths, heights, n)
        b = np.full(n, block_height)

        for center, width in zip(centers, widths):
            depth = block_height * rng.uniform(0.55, 0.95)
            b -= gaussian(x, center, width * rng.uniform(0.65, 1.10), depth)

    elif scenario == "multiple_correspondences":
        # Independent shifts and extra peaks create ambiguous matches.
        n_repeated = max(2, n_peaks)
        centers = draw_centers(rng, n_repeated, location_range)
        widths = draw_widths(rng, n_repeated, width_range)
        heights = draw_heights(rng, n_repeated, signal_level, signal_variation)

        a = peak_train(x, centers, widths, heights, n)
        b_centers = shifted_centers(rng, centers, displacement_range, n)

        n_extra = draw_count(rng, unrelated_peak_count_range)
        extra_centers = draw_centers(rng, n_extra, location_range)
        extra_widths = draw_widths(rng, n_extra, width_range)
        extra_heights = draw_heights(rng, n_extra, signal_level, signal_variation)

        b = peak_train(x, np.r_[b_centers, extra_centers],
                       np.r_[widths, extra_widths],
                       np.r_[heights * rng.uniform(0.75, 1.05, n_repeated), extra_heights], n)

    elif scenario == "tie":
        # Two coherent copies in B create two nearly equal displacement solutions.
        left_displacement = rng.uniform(*tie_displacement_range)
        right_displacement = rng.uniform(*tie_displacement_range)
        tie_location_range = (max(location_range[0], left_displacement),
                              min(location_range[1], n - 1 - right_displacement))

        if tie_location_range[0] >= tie_location_range[1]:
            raise ValueError("tie_displacement_range leaves no valid peak locations.")

        centers = draw_centers(rng, n_peaks, tie_location_range)
        widths = draw_widths(rng, n_peaks, width_range)
        heights = draw_heights(rng, n_peaks, signal_level, signal_variation)
        left_heights = heights * rng.uniform(1 - tie_strength_variation, 1 + tie_strength_variation, n_peaks)
        right_heights = heights * rng.uniform(1 - tie_strength_variation, 1 + tie_strength_variation, n_peaks)

        a = peak_train(x, centers, widths, heights, n)
        b = peak_train(x, np.r_[centers - left_displacement, centers + right_displacement],
                       np.r_[widths, widths], np.r_[left_heights, right_heights], n)

    elif scenario == "custom":
        # Independent custom-shape pairs with per-feature shifts.
        a = np.zeros(n)
        b = np.zeros(n)
        b_centers = shifted_centers(rng, centers, displacement_range, n)

        for center_a, center_b, width, height in zip(centers, b_centers, widths, heights):
            a += primitive(x, shape_a, center_a, width, height, n)
            b += primitive(x, shape_b, center_b, width, height * rng.uniform(0.85, 1.15), n)

        a -= 0.06 * max(0, n_peaks - 1)
        b -= 0.06 * max(0, n_peaks - 1)

    else:
        raise ValueError(f"unknown scenario: {scenario}")

    a = add_noise(a, rng, scenario_noise, n)
    b = add_noise(b, rng, scenario_noise, n)

    # Prevent negative signal without flattening legitimate values above 1.
    return np.maximum(a, 0), np.maximum(b, 0)


def plot_simulated_islands(result, title=None):
    matrix = result["matrix"]
    signal_a = result["signal_a"]
    signal_b = result["signal_b"]
    labels = result["island_labels"]
    islands = result["islands"]
    modes = result["modes"]
    n = len(signal_a)

    fig = plt.figure(figsize=(10.5, 10))
    grid = fig.add_gridspec(3, 3, width_ratios=(1, 4, 0.18), height_ratios=(1.2, 1, 4), wspace=0.10, hspace=0.10)
    fig.subplots_adjust(right=0.78)
    ax = fig.add_subplot(grid[2, 1])
    ax_overlay = fig.add_subplot(grid[0, 1], sharex=ax)
    ax_top = fig.add_subplot(grid[1, 1], sharex=ax)
    ax_left = fig.add_subplot(grid[2, 0], sharey=ax)
    ax_colorbar = fig.add_subplot(grid[2, 2])

    image = ax.imshow(matrix, origin="lower", aspect="equal")
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], "--", color="white", lw=1)
    ax.set(xlim=(-0.5, n - 0.5), ylim=(-0.5, n - 0.5), xlabel="B position", ylabel="A position")

    island_lookup = {island["island_id"]: island for island in islands}
    mode_colors = ["#ff2d2d", "#ff8c00", "#ffcc00", "#d81b60", "#b23aee",
                   "#ff5e00", "#c1121f", "#ff4d8d", "#e76f00", "#8b1e3f"]
    selected_island_ids = {island_id for mode in modes for island_id in mode["islands"]}
    handles = []

    # Keep every detected ridge visible for displacement diagnostics.
    for island in islands:
        if island["island_id"] in selected_island_ids:
            continue
        axis_x, axis_y = island["ridge_B"], island["ridge_A"]
        if len(axis_x) == 1:
            ax.plot(axis_x, axis_y, "o", color="black", ms=5)
            ax.plot(axis_x, axis_y, "o", color="#eeeeee", ms=3)
        else:
            ax.plot(axis_x, axis_y, color="black", lw=3)
            ax.plot(axis_x, axis_y, color="#eeeeee", lw=1.3)

    if len(selected_island_ids) < len(islands):
        handles.append(ax.plot([], [], color="#eeeeee", lw=1.3, label="unselected ridge")[0])

    for color_index, mode in enumerate(modes):
        color = mode_colors[color_index % len(mode_colors)]
        for island_id in mode["islands"]:
            mask = labels == island_id
            if mask.any() and (~mask).any():
                ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=2, origin="lower")
            island = island_lookup[island_id]
            axis_x, axis_y = island["ridge_B"], island["ridge_A"]
            if len(axis_x) == 1:
                ax.plot(axis_x, axis_y, "o", color="black", ms=5)
                ax.plot(axis_x, axis_y, "o", color=color, ms=3)
            else:
                ax.plot(axis_x, axis_y, color="black", lw=3)
                ax.plot(axis_x, axis_y, color=color, lw=1.3)
        handles.append(ax.plot([], [], color=color, lw=2,
                               label=f"mode {mode['mode_id']} (lag {mode['signed_lag']:+.1f})")[0])

    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.28, 1))

    positions = np.arange(n)
    track_colors = {"A": "C0", "B": "C1"}
    ax_overlay.plot(positions, signal_a, label="A", color=track_colors["A"], lw=1.5)
    ax_overlay.plot(positions, signal_b, label="B", color=track_colors["B"], lw=1.5)
    ax_overlay.set_ylabel("Overlay")
    ax_overlay.legend(loc="upper right")
    ax_overlay.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_top.plot(positions, signal_b, color=track_colors["B"])
    ax_top.set_ylabel("B")
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_left.plot(signal_a, positions, color=track_colors["A"])
    ax_left.invert_xaxis()
    ax_left.set_xlabel("A")
    ax_left.tick_params(axis="y", left=False, labelleft=False)

    fig.colorbar(image, cax=ax_colorbar, label="Shared signal")
    fig.canvas.draw()
    matrix_position = ax.get_position()
    for track_axis in (ax_overlay, ax_top):
        track_position = track_axis.get_position()
        track_axis.set_position([matrix_position.x0, track_position.y0, matrix_position.width, track_position.height])
    if title is not None:
        fig.suptitle(title)
    plt.show()
    return fig, {"matrix": ax, "signal_a": ax_left, "signal_b": ax_top, "overlay": ax_overlay}


def inspect_simulated_pair(signal_a, signal_b, seed_threshold_fraction=0.4, growth_threshold_fraction=0.39, max_weak_distance=2, gap_tolerance=1, shape_balance_power=1.05, ridge_fraction=0.9, ridge_smoothing=4.0, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5, zero_lag_threshold=5, displacement_fold_threshold=2.0, mode_lag_tolerance=5, mode_score_fraction=0.33, title=None, plot=True):
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)

    if signal_a.ndim != 1 or signal_b.ndim != 1:
        raise ValueError("signal_a and signal_b must be one-dimensional.")
    if len(signal_a) != len(signal_b):
        raise ValueError("signal_a and signal_b must have the same length.")

    matrix = correspondence_matrix(signal_a[None, None, :], signal_b[None, None, :],
                                   core_size=len(signal_a))[0]
    seed_mask, island_mask, labels, islands = correspondence_islands(
        matrix, seed_threshold_fraction=seed_threshold_fraction,
        growth_threshold_fraction=growth_threshold_fraction,
        max_weak_distance=max_weak_distance, gap_tolerance=gap_tolerance,
        shape_balance_power=shape_balance_power, ridge_fraction=ridge_fraction,
        ridge_smoothing=ridge_smoothing,
        split_prominence_fraction=split_prominence_fraction,
        split_min_peak_fraction=split_min_peak_fraction,
        split_min_peak_distance=split_min_peak_distance,
        split_smoothing=split_smoothing, split_min_seed_area=split_min_seed_area)
    conflicts = island_conflicts(labels, islands)
    selection = select_correspondence_modes(
        islands, conflicts, zero_lag_threshold=zero_lag_threshold,
        displacement_fold_threshold=displacement_fold_threshold,
        mode_lag_tolerance=mode_lag_tolerance, mode_score_fraction=mode_score_fraction)

    result = {
        "signal_a": signal_a,
        "signal_b": signal_b,
        "matrix": matrix,
        "seed_mask": seed_mask,
        "island_mask": island_mask,
        "island_labels": labels,
        "islands": islands,
        "conflicts": conflicts,
        "selection": selection,
        "modes": selection["modes"],
        "island_modes": selection["island_modes"],
    }

    if plot:
        result["figure"], result["axes"] = plot_simulated_islands(result, title=title)

    return result
