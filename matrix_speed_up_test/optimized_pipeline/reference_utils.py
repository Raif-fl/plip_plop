from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy import ndimage

def calculate_rpm_pseudocount(library_sizes):
    """
    Calculates an RPM scale pseudocount that is the equivalent to adding a single read pseudocount to raw data. 
    """
    library_sizes = np.atleast_1d(library_sizes).astype(float)
    return np.mean(1e6 / library_sizes)

def abundance_weighted_enrichment(ip, in_signal, ip_pseudocount, in_pseudocount, mean_core_signal, p=2, smoothing_sigma=None, smoothing_mode="reflect", smoothing_truncate=4.0):
    """
    Calculate an abundance-weighted IP/IN enrichment score at each position.

    Similar to the original "relative information" score used in the lab.
    """
    
    # Calculate a basic log enrichment score.
    log_enrichment = np.log((ip + ip_pseudocount) / (in_signal + in_pseudocount))
    log_enrichment = np.maximum(log_enrichment, 0)

    # Calculate a pseudo-shrinkage factor for the abundance weight.
    k = p * mean_core_signal

    # Calculate the abundance weight from the IP signal at each position.
    denominator = ip + k
    abundance_weight = np.divide(ip, denominator, out=np.zeros_like(ip, dtype=float), where=denominator > 0)

    # Calculate the final enrichment score.
    score = log_enrichment * abundance_weight

    # Smooth the final score across positions.
    if smoothing_sigma is not None and smoothing_sigma > 0:
        score = gaussian_filter1d(score, sigma=smoothing_sigma, axis=-1, mode=smoothing_mode, truncate=smoothing_truncate)

    return score

def correspondence_matrix(signal_A, signal_B):
    """
    Build a correspondence matrix for each 300-nt window.

    Each cell M[i, j] contains min(A_i, B_j).
    """

    # Average replicates at each position.
    A = signal_A.mean(axis=1)
    B = signal_B.mean(axis=1)

    # Compare every position in A with every position in B.
    return np.minimum(A[:, :, None], B[:, None, :]).astype(np.float32)

def island_central_ridge(matrix, region, ridge_fraction=0.9):
    """
    Identify the strongest continuous region of an island and draw a ridge
    through its geometric center, following its longest dimension.
    """

    ####################### Identify the super-island #######################

    # Find the maximum signal within the original island.
    peak = matrix[region].max()

    # Identify all pixels within the requested fraction of the maximum signal.
    super_region = region & (matrix >= ridge_fraction * peak)

    # Identify separate connected regions above the threshold.
    # Pixels touching horizontally, vertically, or diagonally are connected.
    labels, _ = ndimage.label(super_region, structure=np.ones((3, 3)))

    # If multiple super-islands exist, select the one with the most pixels.
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    super_region = labels == np.argmax(sizes)

    ####################### Determine ridge orientation #######################

    # Identify the dimensions of the super-island.
    coords = np.argwhere(super_region)
    A_min, A_max = coords[:, 0].min(), coords[:, 0].max()
    B_min, B_max = coords[:, 1].min(), coords[:, 1].max()
    A_width, B_width = A_max - A_min + 1, B_max - B_min + 1

    # Find the geometric center of the super-island's bounding box.
    A_center = (A_min + A_max) / 2
    B_center = (B_min + B_max) / 2

    ####################### Construct the ridge #######################

    # If the super-island is taller than it is wide, draw a vertical ridge.
    if A_width > B_width:
        long_axis = np.arange(A_min, A_max + 1, dtype=float)
        ridge = np.column_stack((long_axis, np.full(len(long_axis), B_center)))

    # Otherwise, draw a horizontal ridge (including perfect squares).
    else:
        long_axis = np.arange(B_min, B_max + 1, dtype=float)
        ridge = np.column_stack((np.full(len(long_axis), A_center), long_axis))

    return ridge

def island_axis_profile(matrix, region, axis):
    """
    Return a maximum support profile along the A or B axis.
    A maximum support profile is basically just a line filled with all of the maximum values along an axis. 
    """
    # Identify profile along A or B axis. 
    if axis == "A":
        positions = np.flatnonzero(region.any(axis=1))
        profile = np.array([matrix[position, region[position]].max() for position in positions])
    elif axis == "B":
        positions = np.flatnonzero(region.any(axis=0))
        profile = np.array([matrix[region[:, position], position].max() for position in positions])

    return positions, profile

def best_island_split(matrix, region, seed_mask, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_min_seed_area=5):
    """
    Find the strongest valid island-local valley along either axis.
    """
    candidates = []

    for axis in ("A", "B"):
        # Identify the maximum support profile and record its positions. 
        positions, profile = island_axis_profile(matrix, region, axis)
        
        # Reject any islands too small to evaluate. 
        if len(profile) < 2 * split_min_peak_distance + 1:
            continue

        # identify simple, mathematical peaks in the profile.  
        peaks, _ = find_peaks(profile, distance=split_min_peak_distance)

        # Ensure at least 2 peaks (cannot do a valley based split with just 1). 
        if len(peaks) < 2:
            continue

        # Remove extremely weak peaks relative to other peaks in the island. 
        peaks = peaks[profile[peaks] >= split_min_peak_fraction * profile.max()]

        # Go through all valid peak combinations that can contain valleys. 
        for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):

            # Identify the valley. 
            valley_segment = profile[left_peak:right_peak + 1]
            valley = left_peak + int(np.argmin(valley_segment))

            # Determine if the valley is deep enough to seperate the island. 
            smaller_peak = min(profile[left_peak], profile[right_peak])
            valley_depth = 1 - profile[valley] / smaller_peak if smaller_peak > 0 else 0.0
            if valley_depth < split_prominence_fraction:
                continue

            # Make a proposed "cut" for the island seperation. 
            cut = int(positions[valley])
            coordinates = np.indices(region.shape)[0 if axis == "A" else 1]
            child_1 = region & (coordinates <= cut)
            child_2 = region & (coordinates > cut)
            seed_area_1 = int(np.count_nonzero(seed_mask & child_1))
            seed_area_2 = int(np.count_nonzero(seed_mask & child_2))
            if seed_area_1 < split_min_seed_area or seed_area_2 < split_min_seed_area:
                continue

            candidates.append({
                "valley_depth": float(valley_depth), 
                "smaller_peak": float(smaller_peak), 
                "children": (child_1, child_2),
            })

    # Select and return the best proposed split (if any). 
    return max(candidates, key=lambda x: (x["valley_depth"], x["smaller_peak"])) if candidates else None

def split_island_region(matrix, region, seed_mask, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_min_seed_area=5):
    """Recursively split an overmerged island at its strongest valid axis valley."""
    pending = [region]
    finished = []

    while pending:
        current = pending.pop()
        split = best_island_split(matrix, current, seed_mask, split_prominence_fraction=split_prominence_fraction,
                                  split_min_peak_fraction=split_min_peak_fraction, split_min_peak_distance=split_min_peak_distance,
                                  split_min_seed_area=split_min_seed_area)
        if split is None:
            finished.append(current)
        else:
            pending.extend(split["children"])

    return finished


def correspondence_islands(matrix, seed_threshold_fraction=0.4, growth_threshold_fraction=0.39, max_weak_distance=2, gap_tolerance=1, shape_balance_power=1.05, ridge_fraction=0.9, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_min_seed_area=5):
    """
    Identify correspondence-support islands. 
    """

    # Extra max value. 
    max_value = matrix.max()

    # Handle the 0 signal case (no reason to keep going here). 
    if max_value <= 0:
        empty_bool = np.zeros_like(matrix, dtype=bool)
        empty_labels = np.zeros_like(matrix, dtype=np.int32)

        return empty_bool, empty_bool, empty_labels, []

    # Redefine the seed and growth thresholds according to the maximum value. 
    seed_threshold = seed_threshold_fraction * max_value
    growth_threshold = growth_threshold_fraction * max_value

    # Strong support required to initiate an island.
    seed_mask = matrix > seed_threshold

    # Weaker support that may be incorporated into an island.
    growth_mask = matrix > growth_threshold

    # Tolerate very short interruptions below the growth threshold.
    if gap_tolerance > 0:
        structure = np.ones((3, 3), dtype=bool)
        growth_mask = ndimage.binary_closing(growth_mask,structure=structure,iterations=gap_tolerance)

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

    # Use 8 connectivity to combine signal masks into actual candidate islands. 
    structure = np.ones((3, 3), dtype=bool)
    raw_labels, n_regions = ndimage.label(island_mask, structure=structure)

    # Additional splitting used to differentiate merged peaks. 
    split_regions = []
    for raw_id in range(1, n_regions + 1):
        region = raw_labels == raw_id

        # Only retain regions that contain actual strong seed support.
        if not np.any(seed_mask & region):
            continue

        # Perform the splitting. 
        split_regions.extend(split_island_region(matrix, region, seed_mask, split_prominence_fraction=split_prominence_fraction,
                                                 split_min_peak_fraction=split_min_peak_fraction, split_min_peak_distance=split_min_peak_distance,
                                                split_min_seed_area=split_min_seed_area))

    # Initialize list for island creation. 
    labels = np.zeros_like(raw_labels, dtype=np.int32)
    islands = []

    # Loop through all split regions and convert them to proper islands.
    for island_id, region in enumerate(split_regions, start=1):
    
        # Set island ID.
        labels[region] = island_id
    
        # Calculate island width in both directions.
        coords = np.argwhere(region)
        A_positions = coords[:, 0]
        B_positions = coords[:, 1]
        A_width = A_positions.max() - A_positions.min() + 1
        B_width = B_positions.max() - B_positions.min() + 1
    
        # Calculate total correspondence signal across the island.
        values = matrix[A_positions, B_positions]
        total_support = values.sum()
    
        # Penalize strongly oblong islands when calculating the selection score.
        shape_balance = min(A_width, B_width) / max(A_width, B_width)
        selection_score = total_support * shape_balance ** shape_balance_power
    
        # Calculate the ridge and its possible lag interval.
        ridge = island_central_ridge(matrix, region, ridge_fraction=ridge_fraction)
        ridge_lags = ridge[:, 1] - ridge[:, 0]
    
        island = {
            "island_id": island_id,
            "total_support": total_support,
            "selection_score": selection_score,
            "ridge": ridge,
            "lag_min": float(ridge_lags.min()),
            "lag_max": float(ridge_lags.max()),
        }
    
        islands.append(island)
    
    return seed_mask, island_mask, labels, islands

def constrain_island_to_lag(matrix, labels, island, cell_lags, lag, shape_balance_power):
    """
    Restrict one island to a single shared lag for scoring and conflicts.
    """

    # Select the island cells.
    full_region = labels == island["island_id"]

    # Weight the two diagonals nearest the requested lag.
    line_weights = np.maximum(1 - np.abs(cell_lags - lag), 0)

    # Extract out region where line weights are above 0. 
    region = full_region & (line_weights > 0)

    # Extract out coordinates of valid region. 
    coordinates = np.argwhere(region)
    # An allowed lag may pass through a gap in the island.
    if len(coordinates) == 0:
        return {
            "island_id": island["island_id"], "region": region,
            "ridge": np.empty((0, 2), dtype=float),
            "total_support": 0.0, "selection_score": 0.0,
        }

    A_positions = coordinates[:, 0]
    B_positions = coordinates[:, 1]

    # Get new widths and support score. 
    A_width = A_positions.max() - A_positions.min() + 1
    B_width = B_positions.max() - B_positions.min() + 1
    total_support = np.sum(matrix[region] * line_weights[region])
    shape_balance = min(A_width, B_width) / max(A_width, B_width)
    selection_score = total_support * shape_balance ** shape_balance_power

    # Keep ridge points within half a cell of the selected lag.
    ridge = island["ridge"]
    ridge_lags = ridge[:, 1] - ridge[:, 0]
    ridge = ridge[np.abs(ridge_lags - lag) <= 0.5]

    return {
        "island_id": island["island_id"],
        "region": region,
        "ridge": ridge,
        "total_support": total_support,
        "selection_score": selection_score,
    }

def constrain_island_to_mode_interval(labels, island, cell_lags, lag_min, lag_max):
    """Return the full island area reachable anywhere within a mode interval."""

    # Select island cells within half a cell of the lag interval.
    full_region = labels == island["island_id"]
    region = full_region & (cell_lags >= lag_min - 0.5) & (cell_lags <= lag_max + 0.5)

    # Keep ridge points within half a cell of the lag interval.
    ridge = island["ridge"]
    ridge_lags = ridge[:, 1] - ridge[:, 0]
    ridge = ridge[(ridge_lags >= lag_min - 0.5) & (ridge_lags <= lag_max + 0.5)]

    return {"island_id": island["island_id"], "region": region, "ridge": ridge}

def region_bounds(region):
    """Return one constrained region's A_min, A_max, B_min, and B_max."""
    coordinates = np.argwhere(region)
    return (int(coordinates[:, 0].min()), int(coordinates[:, 0].max()),
            int(coordinates[:, 1].min()), int(coordinates[:, 1].max()))

def regions_conflict(region_1, region_2):
    """Return whether two constrained regions reuse any A or B positions."""
    A_overlap = np.any(region_1.any(axis=1) & region_2.any(axis=1))
    B_overlap = np.any(region_1.any(axis=0) & region_2.any(axis=0))
    return A_overlap or B_overlap

def strongest_mode_compatible_subset(members):
    """Find the strongest position-preserving subset of constrained members."""

    # Attach region bounds and sort members by their starting positions.
    bounded = [(member, *region_bounds(member["region"])) for member in members if member["region"].any()]
    if not bounded:
        return (), 0.0

    ordered = sorted(bounded, key=lambda x: (x[1], x[3], x[0]["island_id"]))
    best_scores = []
    best_paths = []

    # Build the highest-scoring compatible chain ending at each member.
    for j, (member, member_A_min, _, member_B_min, _) in enumerate(ordered):
        best_score = member["selection_score"]
        best_path = (member["island_id"],)

        for i, (_, _, previous_A_max, _, previous_B_max) in enumerate(ordered[:j]):
            if previous_A_max >= member_A_min or previous_B_max >= member_B_min:
                continue

            candidate_score = best_scores[i] + member["selection_score"]
            candidate_path = best_paths[i] + (member["island_id"],)
            if (candidate_score, len(candidate_path)) > (best_score, len(best_path)):
                best_score = candidate_score
                best_path = candidate_path

        best_scores.append(best_score)
        best_paths.append(best_path)

    best_index = max(range(len(ordered)), key=lambda i: (best_scores[i], len(best_paths[i])))
    return best_paths[best_index], best_scores[best_index]

def score_correspondence_mode_at_lag(members, matrix, labels, cell_lags, shape_balance_power, lag):
    """Constrain every member to one shared lag and score the resulting mode."""

    # Constrain every member to the requested lag.
    constrained_members = [
        constrain_island_to_lag(matrix, labels, island, cell_lags, lag, shape_balance_power)
        for island in members
    ]

    # Select the highest-scoring chain with non-overlapping A and B ranges.
    compatible_islands, compatible_score = strongest_mode_compatible_subset(constrained_members)
    compatible_ids = set(compatible_islands)
    region_conflict_pairs = tuple(
        (constrained_members[i]["island_id"], constrained_members[j]["island_id"])
        for i in range(len(constrained_members)) for j in range(i + 1, len(constrained_members))
        if regions_conflict(constrained_members[i]["region"], constrained_members[j]["region"])
    )
    possible_pairs = len(constrained_members) * (len(constrained_members) - 1) / 2
    candidate_ids = tuple(sorted(member["island_id"] for member in constrained_members))
    excluded_ids = tuple(island_id for island_id in candidate_ids if island_id not in compatible_ids)

    return {
        "candidate_islands": candidate_ids,
        "islands": compatible_islands,
        "excluded_islands": excluded_ids,
        "n_candidate_islands": len(candidate_ids),
        "n_islands": len(compatible_islands),
        "compatible_islands": compatible_islands,
        "n_independent_islands": len(compatible_islands),
        "region_conflicts": region_conflict_pairs,
        "conflict_fraction": len(region_conflict_pairs) / possible_pairs if possible_pairs else 0.0,
        "lag": float(lag),
        "selection_score": compatible_score,
        "all_island_score": sum(member["selection_score"] for member in constrained_members),
        "total_support": sum(member["total_support"] for member in constrained_members),
        "compatible_total_support": sum(member["total_support"] for member in constrained_members
                                        if member["island_id"] in compatible_ids),
        "scoring_regions": {member["island_id"]: member["region"] for member in constrained_members},
        "member_selection_scores": {member["island_id"]: member["selection_score"]
                                    for member in constrained_members},
    }

def candidate_mode_lags(lag_min, lag_max):
    """Return every lag at which a mode's line score can reach a maximum."""

    # Combine the interval endpoints with all enclosed integer lags.
    integer_lags = np.arange(np.ceil(lag_min), np.floor(lag_max) + 1)
    return np.unique(np.concatenate(([lag_min], integer_lags, [lag_max]))).astype(float)

def build_correspondence_mode(members, matrix, labels, shape_balance_power, lag_min, lag_max):
    """Move one shared line through a mode interval and retain its best lag."""

    # Calculate the lag at each matrix position.
    A_coordinates, B_coordinates = np.indices(matrix.shape)
    cell_lags = B_coordinates - A_coordinates

    # Score the member set at each candidate lag.
    scored_lags = [
        score_correspondence_mode_at_lag(members, matrix, labels, cell_lags, shape_balance_power, lag)
        for lag in candidate_mode_lags(lag_min, lag_max)
    ]
    # Only supported lags can represent a mode.
    supported_lags = [candidate for candidate in scored_lags if candidate["n_islands"] > 0]
    if not supported_lags:
        return None

    # Break score ties by member count, total evidence, then interval proximity.
    interval_center = (lag_min + lag_max) / 2
    mode = max(supported_lags, key=lambda x: (
        x["selection_score"], x["n_islands"], x["all_island_score"],
        -abs(x["lag"] - interval_center)))

    mode["lag_min"] = lag_min
    mode["lag_max"] = lag_max
    mode["zero_lag"] = lag_min <= 0 <= lag_max
    mode["evaluated_lags"] = tuple(candidate["lag"] for candidate in scored_lags)

    # Build interval-constrained regions and ridges for each member.
    interval_members = [
        constrain_island_to_mode_interval(labels, island, cell_lags, lag_min, lag_max)
        for island in members
    ]
    mode["constrained_regions"] = {member["island_id"]: member["region"] for member in interval_members}
    mode["constrained_ridges"] = {member["island_id"]: member["ridge"] for member in interval_members}
    return mode

def summarize_correspondence_mode(members, matrix, labels, shape_balance_power):
    """Find the common lag interval and build its mode-constrained regions."""
    lag_min = max(island["lag_min"] for island in members)
    lag_max = min(island["lag_max"] for island in members)
    return build_correspondence_mode(members, matrix, labels, shape_balance_power, lag_min, lag_max)

def number_modes(modes):
    """Sort modes by representative lag and assign sequential IDs."""

    numbered = []
    for mode_id, mode in enumerate(sorted(modes, key=lambda x: (x["lag"], x["lag_min"], x["lag_max"])), start=1):
        numbered_mode = dict(mode)
        numbered_mode["mode_id"] = mode_id
        numbered.append(numbered_mode)
    return numbered

def correspondence_modes(islands, matrix, labels, shape_balance_power):
    """Find maximal island groups with overlapping ridge-lag intervals."""

    # Collect all ridge-interval endpoints.
    endpoints = sorted({value for island in islands for value in (island["lag_min"], island["lag_max"])})

    # Build the island set covering each endpoint.
    candidate_sets = {
        frozenset(island["island_id"] for island in islands
                  if island["lag_min"] <= point <= island["lag_max"])
        for point in endpoints
    }

    # Remove sets contained within larger candidate sets.
    maximal_sets = [candidate for candidate in candidate_sets if not any(candidate < other for other in candidate_sets)]

    # Index islands by ID.
    island_lookup = {island["island_id"]: island for island in islands}

    # Build each maximal set over its shared lag interval.
    modes = [
        summarize_correspondence_mode(
            [island_lookup[island_id] for island_id in sorted(group)],
            matrix, labels, shape_balance_power)
        for group in maximal_sets
    ]
    return number_modes([mode for mode in modes if mode is not None])

def modes_conflict(mode_1, mode_2):
    """Return whether two modes reuse A or B signal at their selected lags."""
    for island_1 in mode_1["compatible_islands"]:
        region_1 = mode_1["scoring_regions"][island_1]
        for island_2 in mode_2["compatible_islands"]:
            region_2 = mode_2["scoring_regions"][island_2]
            if regions_conflict(region_1, region_2):
                return True
    return False

def remove_mode_members_conflicting_with_reference(mode, reference_mode, island_lookup, matrix, labels, shape_balance_power):
    """Remove only the portions of a mode that compete with a retained mode."""
    reference_regions = [
        reference_mode["scoring_regions"][island_id]
        for island_id in reference_mode["compatible_islands"]
    ]
    remaining_ids = []
    removed_ids = []

    # Partition candidate islands by conflict with the reference regions.
    for island_id in mode["candidate_islands"]:
        region = mode["scoring_regions"][island_id]
        if any(regions_conflict(region, reference_region) for reference_region in reference_regions):
            removed_ids.append(island_id)
        else:
            remaining_ids.append(island_id)

    if not remaining_ids:
        return None, tuple(removed_ids)

    # Rebuild the remaining members over the original lag interval.
    remaining_members = [island_lookup[island_id] for island_id in remaining_ids]
    trimmed_mode = build_correspondence_mode(
        remaining_members, matrix, labels, shape_balance_power,
        mode["lag_min"], mode["lag_max"])
    return trimmed_mode, tuple(removed_ids)

def select_correspondence_modes(islands, matrix, labels, shape_balance_power, displacement_fold_threshold=2.0, mode_score_fraction=0.33):
    """
    Form modes, resolve central-mode conflicts, and retain viable modes.
    """

    # Build all provisional modes.
    provisional_modes = correspondence_modes(islands, matrix, labels, shape_balance_power)

    # Return an empty selection when no provisional modes exist.
    if not provisional_modes:
        return {
            "surviving_islands": [], "deleted_island_ids": set(),
            "provisional_modes": [], "candidate_modes": [], "modes": [],
            "central_gate": {"status": "absent"},
            "rejected_provisional_mode_ids": set(),
            "rejected_mode_ids": set(), "island_modes": {},
            "mode_conflicts": [], "removed_mode_members": [],
        }

    # Normalize provisional scores to the strongest score.
    strongest_score = max(mode["selection_score"] for mode in provisional_modes)
    for mode in provisional_modes:
        mode["score_fraction_of_strongest"] = mode["selection_score"] / strongest_score if strongest_score > 0 else 0.0

    # Find the provisional mode whose interval contains zero lag.
    central_mode = next((mode for mode in provisional_modes if mode["zero_lag"]), None)

    removed_mode_members = []
    rejected_provisional_mode_ids = set()
    central_gate = {"status": "absent"}
    candidate_modes = [dict(mode) for mode in provisional_modes]

    # Record pairwise conflicts among provisional modes.
    mode_conflicts = [
        (provisional_modes[i]["mode_id"], provisional_modes[j]["mode_id"])
        for i in range(len(provisional_modes)) for j in range(i + 1, len(provisional_modes))
        if modes_conflict(provisional_modes[i], provisional_modes[j])
    ]

    # Resolve conflicts when a zero-lag mode exists.
    if central_mode is not None:
        # Collect displaced modes that conflict with the central mode.
        conflicting_modes = [
            mode for mode in provisional_modes
            if not mode["zero_lag"] and modes_conflict(central_mode, mode)
        ]

        # Select the strongest conflicting displaced mode.
        strongest_conflicting = max(conflicting_modes, key=lambda x: x["selection_score"]) if conflicting_modes else None

        # Reject a central mode below the minimum score fraction.
        if not central_mode["score_fraction_of_strongest"] >= mode_score_fraction:
            rejected_provisional_mode_ids.add(central_mode["mode_id"])
            candidate_modes = [dict(mode) for mode in provisional_modes if not mode["zero_lag"]]
            central_gate = {
                "status": "rejected", "reason": "below_mode_score_fraction",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_lag_interval": (central_mode["lag_min"], central_mode["lag_max"]),
                "central_compatible_islands": central_mode["compatible_islands"],
                "score_fraction_of_strongest": central_mode["score_fraction_of_strongest"],
            }

        # Reject a central mode overwhelmed by a conflicting displaced mode.
        elif strongest_conflicting is not None and strongest_conflicting["selection_score"] >= displacement_fold_threshold * central_mode["selection_score"]:
            rejected_provisional_mode_ids.add(central_mode["mode_id"])
            candidate_modes = [dict(mode) for mode in provisional_modes if not mode["zero_lag"]]
            central_gate = {
                "status": "rejected", "reason": "overwhelmed_by_displaced_mode",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_lag_interval": (central_mode["lag_min"], central_mode["lag_max"]),
                "central_compatible_islands": central_mode["compatible_islands"],
                "displaced_mode_id": strongest_conflicting["mode_id"],
                "displaced_score": strongest_conflicting["selection_score"],
                "required_displaced_score": displacement_fold_threshold * central_mode["selection_score"],
            }
        # Otherwise trim conflicting members from displaced modes.
        else:
            island_lookup = {island["island_id"]: island for island in islands}
            candidate_modes = [dict(central_mode)]

            for mode in provisional_modes:
                if mode["zero_lag"]:
                    continue

                trimmed_mode, removed_ids = remove_mode_members_conflicting_with_reference(
                    mode, central_mode, island_lookup, matrix, labels, shape_balance_power)
                if removed_ids:
                    removed_mode_members.append({
                        "provisional_mode_id": mode["mode_id"],
                        "island_ids": removed_ids,
                    })
                if trimmed_mode is not None:
                    candidate_modes.append(trimmed_mode)

            central_gate = {
                "status": "retained", "reason": "central_mode_won_conflicts",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_lag_interval": (central_mode["lag_min"], central_mode["lag_max"]),
                "central_compatible_islands": central_mode["compatible_islands"],
                "strongest_conflicting_mode_id": strongest_conflicting["mode_id"] if strongest_conflicting else None,
                "strongest_conflicting_score": strongest_conflicting["selection_score"] if strongest_conflicting else None,
            }

    # Retain islands represented by at least one candidate mode.
    surviving_ids = {island_id for mode in candidate_modes for island_id in mode["islands"]}
    surviving_islands = [island for island in islands if island["island_id"] in surviving_ids]
    deleted_island_ids = {island["island_id"] for island in islands} - surviving_ids
    # Renumber candidate modes and normalize their scores.
    candidate_modes = number_modes(candidate_modes)

    strongest_score = max((mode["selection_score"] for mode in candidate_modes), default=0.0)

    for mode in candidate_modes:
        mode["score_fraction_of_strongest"] = mode["selection_score"] / strongest_score if strongest_score > 0 else 0.0

    # Keep candidates above the score-fraction threshold.
    retained_candidates = [mode for mode in candidate_modes
                           if mode["score_fraction_of_strongest"] >= mode_score_fraction]
    rejected_mode_ids = {mode["mode_id"] for mode in candidate_modes if mode not in retained_candidates}

    # Renumber retained modes while preserving their candidate IDs.
    modes = []
    for mode_id, candidate in enumerate(retained_candidates, start=1):
        mode = dict(candidate)
        mode["candidate_mode_id"] = candidate["mode_id"]
        mode["mode_id"] = mode_id
        modes.append(mode)

    # Map each island ID to its retained mode IDs.
    island_modes = {}
    for mode in modes:
        for island_id in mode["islands"]:
            island_modes.setdefault(island_id, []).append(mode["mode_id"])
    island_modes = {island_id: tuple(mode_ids) for island_id, mode_ids in island_modes.items()}

    return {
        "surviving_islands": surviving_islands,
        "deleted_island_ids": deleted_island_ids,
        "provisional_modes": provisional_modes,
        "candidate_modes": candidate_modes,
        "modes": modes,
        "central_gate": central_gate,
        "rejected_provisional_mode_ids": rejected_provisional_mode_ids,
        "rejected_mode_ids": rejected_mode_ids,
        "island_modes": island_modes,
        "mode_conflicts": mode_conflicts,
        "removed_mode_members": removed_mode_members,
    }

def plot_islands(result, title=None):
    matrix = result["matrix"]
    signal_a = result["signal_a"]
    signal_b = result["signal_b"]
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

    mode_colors = ["#ff2d2d", "#ff8c00", "#ffcc00", "#d81b60", "#b23aee",
                   "#ff5e00", "#c1121f", "#ff4d8d", "#e76f00", "#8b1e3f"]
    selected_island_ids = {island_id for mode in modes for island_id in mode["islands"]}
    handles = []

    # Draw selected ridges faintly and unselected ridges with black outlines.
    for island in islands:
        ridge = island["ridge"]
        axis_x, axis_y = ridge[:, 1], ridge[:, 0]
        if island["island_id"] in selected_island_ids:
            if len(axis_x) == 1:
                ax.plot(axis_x, axis_y, "o", color="#dddddd", ms=3)
            else:
                ax.plot(axis_x, axis_y, color="#dddddd", lw=1.0)
        elif len(axis_x) == 1:
            ax.plot(axis_x, axis_y, "o", color="black", ms=5)
            ax.plot(axis_x, axis_y, "o", color="#eeeeee", ms=3)
        else:
            ax.plot(axis_x, axis_y, color="black", lw=3)
            ax.plot(axis_x, axis_y, color="#eeeeee", lw=1.3)

    if len(selected_island_ids) < len(islands):
        handles.append(ax.plot([], [], color="#eeeeee", lw=1.3, label="unselected ridge")[0])

    # Draw each mode's interval-constrained regions and ridges.
    for color_index, mode in enumerate(modes):
        color = mode_colors[color_index % len(mode_colors)]
        for island_id in mode["islands"]:
            region = mode["constrained_regions"][island_id]
            ridge = mode["constrained_ridges"][island_id]

            if region.any() and (~region).any():
                ax.contour(region.astype(float), levels=[0.5], colors=color,
                           linewidths=2, origin="lower")

            if len(ridge) == 0:
                continue
            axis_x, axis_y = ridge[:, 1], ridge[:, 0]
            if len(axis_x) == 1:
                ax.plot(axis_x, axis_y, "o", color="black", ms=5)
                ax.plot(axis_x, axis_y, "o", color=color, ms=3)
            else:
                ax.plot(axis_x, axis_y, color="black", lw=3)
                ax.plot(axis_x, axis_y, color=color, lw=1.3)

        handles.append(ax.plot([], [], color=color, lw=2,
                               label=(f"mode {mode['mode_id']} (selected lag {mode['lag']:+.1f}; "
                                      f"allowed {mode['lag_min']:+.1f} to {mode['lag_max']:+.1f})"))[0])

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

def identify_islands(signal_a, signal_b, seed_threshold_fraction=0.4, growth_threshold_fraction=0.39, max_weak_distance=2, gap_tolerance=1, shape_balance_power=1.05, ridge_fraction=0.9, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_min_seed_area=5, displacement_fold_threshold=2.0, mode_score_fraction=0.33, title=None, plot=True):
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)

    if signal_a.ndim != 1 or signal_b.ndim != 1:
        raise ValueError("signal_a and signal_b must be one-dimensional.")
    if len(signal_a) != len(signal_b):
        raise ValueError("signal_a and signal_b must have the same length.")

    matrix = correspondence_matrix(signal_a[None, None, :], signal_b[None, None, :])[0]
    seed_mask, island_mask, labels, islands = correspondence_islands(matrix, seed_threshold_fraction=seed_threshold_fraction, 
                                                                     growth_threshold_fraction=growth_threshold_fraction,
                                                                     max_weak_distance=max_weak_distance, gap_tolerance=gap_tolerance,
                                                                     shape_balance_power=shape_balance_power, ridge_fraction=ridge_fraction,
                                                                     split_prominence_fraction=split_prominence_fraction,
                                                                     split_min_peak_fraction=split_min_peak_fraction,
                                                                     split_min_peak_distance=split_min_peak_distance, split_min_seed_area=split_min_seed_area)
    
    selection = select_correspondence_modes(islands, matrix, labels, shape_balance_power,displacement_fold_threshold=displacement_fold_threshold,
                                            mode_score_fraction=mode_score_fraction)

    result = {
        "signal_a": signal_a,
        "signal_b": signal_b,
        "matrix": matrix,
        "seed_mask": seed_mask,
        "island_mask": island_mask,
        "island_labels": labels,
        "islands": islands,
        "selection": selection,
        "modes": selection["modes"],
        "island_modes": selection["island_modes"],
        "mode_conflicts": selection["mode_conflicts"],
    }

    if plot:
        result["figure"], result["axes"] = plot_islands(result, title=title)

    return result
