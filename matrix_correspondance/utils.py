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

def abundance_weighted_enrichment(ip, in_signal, ip_pseudocount, in_pseudocount, mean_core_signal, p=2):
    """
    Calculate an IP/IN enrichment score at each position and 

    Similar to the original "relative information" score used in the lab. 
    """
    
    # Calculate a basic log enrichment score. 
    log_enrichment = np.log((ip + ip_pseudocount) /(in_signal + in_pseudocount))
    log_enrichment = np.maximum(log_enrichment, 0)

    # Calculate a sort of pseudo "shrinkage factor" for the abundance weight.  
    k = p * mean_core_signal

    # Calculate the abundance weight as a function of the total window signal captured at each position. 
    denominator = ip + k
    abundance_weight = np.divide(ip, denominator, out=np.zeros_like(ip, dtype=float), where=denominator > 0)

    return log_enrichment * abundance_weight

def correspondence_matrix(signal_A, signal_B, core_size):
    """
    Build a correspondence matrix for each window comparison.

    Each cell M[i, j] contains the min(A_i, B_j).
    """

    # Extracts out the core part of the signal in the window
    ###!!!###Required because we made all the windows extra big for earlier cross-correlation prototype,
    ###!!!### should eventually be removed after changing window definition again
    context = (signal_A.shape[2] - core_size) // 2
    core_start = context
    core_end = context + core_size

    # Central 300-nt window.
    A = signal_A[:, :, core_start:core_end].mean(axis=1)
    B = signal_B[:, :, core_start:core_end].mean(axis=1)

    # Compare every position in A with every position in B.
    matrices = np.minimum(A[:, :, None], B[:, None, :]).astype(np.float32)

    return matrices

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

def island_axis_profile(matrix, region, axis, smoothing):
    """
    Return a maximum support profile along the A or B axis.
    A maximum support profil is basically just a line filled with all of the maximum values along an axis. 
    """
    # Identify profile along A or B axis. 
    if axis == "A":
        positions = np.flatnonzero(region.any(axis=1))
        profile = np.array([matrix[position, region[position]].max() for position in positions])
    elif axis == "B":
        positions = np.flatnonzero(region.any(axis=0))
        profile = np.array([matrix[region[:, position], position].max() for position in positions])

    # Additional smoothing is helpful for identifying splits. 
    smooth = gaussian_filter1d(profile, smoothing, mode="nearest") if smoothing > 0 else profile
    return positions, smooth

def best_island_split(matrix, region, seed_mask, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5):
    """
    Find the strongest valid island-local valley along either axis.
    """
    candidates = []

    for axis in ("A", "B"):
        # Identify the maximum support profile and record its positions. 
        positions, profile = island_axis_profile(matrix, region, axis, split_smoothing)
        
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

def split_island_region(matrix, region, seed_mask, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5):
    """Recursively split an overmerged island at its strongest valid axis valley."""
    pending = [region]
    finished = []

    while pending:
        current = pending.pop()
        split = best_island_split(matrix, current, seed_mask, split_prominence_fraction=split_prominence_fraction,
                                  split_min_peak_fraction=split_min_peak_fraction, split_min_peak_distance=split_min_peak_distance,
                                  split_smoothing=split_smoothing, split_min_seed_area=split_min_seed_area)
        if split is None:
            finished.append(current)
        else:
            pending.extend(split["children"])

    return finished


def correspondence_islands(matrix, seed_threshold_fraction=0.4, growth_threshold_fraction=0.39, max_weak_distance=2, gap_tolerance=1, shape_balance_power=1.05, ridge_fraction=0.9, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5):
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
                                                 split_smoothing=split_smoothing, split_min_seed_area=split_min_seed_area))

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

def island_conflicts(labels, islands):
    """
    Identify competing correspondence islands.

    Two islands conflict if they occupy any of the same A positions
    or any of the same B positions.
    """

    # Initialize projections dictionary. 
    projections = {}

    # Loop through all islands and record projections. 
    for island in islands:
        island_id = island["island_id"]
        region = labels == island_id

        projections[island_id] = {
            # Which A positions are claimed by this island?
            "A": region.any(axis=1),

            # Which B positions are claimed by this island?
            "B": region.any(axis=0),
        }

    # Initialize conflict list. 
    conflicts = []

    # Loop through all island combinations. 
    for i in range(len(islands)):
        id_i = islands[i]["island_id"]
        for j in range(i + 1, len(islands)):
            id_j = islands[j]["island_id"]

            
            A_overlap = np.any(projections[id_i]["A"] & projections[id_j]["A"])
            B_overlap = np.any(projections[id_i]["B"] & projections[id_j]["B"])

            # If no overlap, move on. 
            if not (A_overlap or B_overlap):
                continue

            conflicts.append({
                "island_1": id_i,
                "island_2": id_j
            })

    return conflicts

def ridge_bounds(ridge):
    """
    Return the ridge bounds as A_min, A_max, B_min, B_max.
    """
    return (float(ridge[:, 0].min()), float(ridge[:, 0].max()),
            float(ridge[:, 1].min()), float(ridge[:, 1].max()))

def ridge_conflicts(island_1, island_2):
    """
    Return whether two ridges claim any of the same A or B positions.
    """
    # Get ridge boundaries. 
    A_1_min, A_1_max, B_1_min, B_1_max = ridge_bounds(island_1["ridge"])
    A_2_min, A_2_max, B_2_min, B_2_max = ridge_bounds(island_2["ridge"])

    # Detect if there are any overlaps. 
    A_overlap = max(A_1_min, A_2_min) <= min(A_1_max, A_2_max)
    B_overlap = max(B_1_min, B_2_min) <= min(B_1_max, B_2_max)
    return A_overlap or B_overlap

def strongest_ridge_compatible_subset(islands):
    """
    Find the strongest non-overlapping, position-preserving subset of islands
    within an already constructed mode.
    """

    # Identify the boundaries of each ridge. 
    bounded = [(island, *ridge_bounds(island["ridge"])) for island in islands]

    # Sort islands by where their ridges begin along A, then B.
    # (establishes the order in which compatible ridge chains are built).
    ordered = sorted(bounded, key=lambda x: (x[1], x[3], x[0]["island_id"]))

    # Initialize lists for the best score and best path. 
    best_scores = []
    best_paths = []

    # Treat each island as the possible endpoint of a compatible chain.
    for j, (island, island_A_min, _, island_B_min, _) in enumerate(ordered):

        # Begin with the current island by itself.
        island_score = island["selection_score"]
        best_score = island_score
        best_path = (island["island_id"],)

        # Check whether the current island can extend a chain ending at any previously processed island.
        for i, (_, _, previous_A_max, _, previous_B_max) in enumerate(ordered[:j]):

            # The previous ridge must finish before the current ridge begins along both A and B. Overlap along
            # either axis means that the two islands are competing for some of the same signal.
            if previous_A_max >= island_A_min or previous_B_max >= island_B_min:
                continue

            # Extend the best chain ending at the previous island.
            candidate_score = best_scores[i] + island_score
            candidate_path = best_paths[i] + (island["island_id"],)

            # Prefer the chain with the greater total score.
            # If scores tie, prefer the chain containing more islands.
            if (candidate_score, len(candidate_path)) > (best_score, len(best_path)):
                best_score = candidate_score
                best_path = candidate_path

        # Store the strongest compatible chain ending at the current island.
        best_scores.append(best_score)
        best_paths.append(best_path)

    # The strongest overall chain may end at any island, so select the best result from all possible endpoints. 
    best_index = max(
        range(len(ordered)),
        key=lambda i: (best_scores[i], len(best_paths[i]))
    )

    return best_paths[best_index], best_scores[best_index]

def summarize_correspondence_mode(members):
    """
    Summarize islands that share at least one common ridge lag.
    """
    lag_min = max(island["lag_min"] for island in members)
    lag_max = min(island["lag_max"] for island in members)
    if lag_min > lag_max:
        raise ValueError("Mode members do not share a common lag.")

    scores = [island.get("selection_score", island["total_support"]) for island in members]
    compatible_islands, compatible_score = strongest_ridge_compatible_subset(members)
    compatible_ids = set(compatible_islands)
    ridge_conflict_pairs = tuple(
        (members[i]["island_id"], members[j]["island_id"])
        for i in range(len(members)) for j in range(i + 1, len(members))
        if ridge_conflicts(members[i], members[j])
    )
    possible_pairs = len(members) * (len(members) - 1) / 2
    zero_lag = lag_min <= 0 <= lag_max

    return {
        "islands": tuple(sorted(island["island_id"] for island in members)),
        "n_islands": len(members),
        "compatible_islands": compatible_islands,
        "n_independent_islands": len(compatible_islands),
        "ridge_conflicts": ridge_conflict_pairs,
        "conflict_fraction": len(ridge_conflict_pairs) / possible_pairs if possible_pairs else 0.0,
        "lag_min": lag_min,
        "lag_max": lag_max,
        "lag": 0.0 if zero_lag else (lag_min + lag_max) / 2,
        "zero_lag": zero_lag,
        # Overlapping ridges remain members but cannot add duplicated evidence.
        "selection_score": compatible_score,
        "all_island_score": sum(scores),
        "total_support": sum(island["total_support"] for island in members),
        "compatible_total_support": sum(island["total_support"] for island in members
                                        if island["island_id"] in compatible_ids),
    }

def number_modes(modes):
    """
    Sort modes by their representative lag and assign stable IDs.
    """
    
    numbered = []
    for mode_id, mode in enumerate(sorted(modes, key=lambda x: (x["lag"], x["lag_min"], x["lag_max"])), start=1):
        numbered_mode = dict(mode)
        numbered_mode["mode_id"] = mode_id
        numbered.append(numbered_mode)
    return numbered

def correspondence_modes(islands):
    """
    Find maximal island groups whose ridge-lag intervals share a point.

    Complex math used for a simple idea, if you can draw a line parallel to the 0 lag line that
    cuts through multiple ridges, the islands with those ridges may be part of the same mode. 
    """

    # For efficiency, we test every endpoint for membership (equivalent to testing the whole interval). 
    endpoints = sorted({value for island in islands for value in (island["lag_min"], island["lag_max"])})

    # Identify candidate sets of islands for mode. 
    candidate_sets = {
        frozenset(island["island_id"] for island in islands
                  if island["lag_min"] <= point <= island["lag_max"])
        for point in endpoints
    }

    # Remove candidate modes that are entirely contained within another valid mode.
    maximal_sets = [candidate for candidate in candidate_sets if not any(candidate < other for other in candidate_sets)]

    # Convert to lookup table. 
    island_lookup = {island["island_id"]: island for island in islands}

    # Convert back to complete island objects. 
    modes = [summarize_correspondence_mode([island_lookup[island_id] for island_id in sorted(group)]) for group in maximal_sets]
    return number_modes(modes)

def select_correspondence_modes(islands, conflicts, displacement_fold_threshold=2.0, mode_score_fraction=0.33):
    """
    Form modes, resolve central-mode conflicts, and retain viable modes.
    """

    # Build every lag hypothesis before the central mode resolves conflicts.
    provisional_modes = correspondence_modes(islands)

    # Rare instance where no proper modes are identified. 
    if not provisional_modes:
        return {
            "surviving_islands": [], "deleted_island_ids": set(),
            "provisional_modes": [], "candidate_modes": [], "modes": [],
            "central_gate": {"status": "absent"},
            "rejected_provisional_mode_ids": set(),
            "rejected_mode_ids": set(), "island_modes": {},
        }

    # # Find the strongest scroing more. 
    strongest_score = max(mode["selection_score"] for mode in provisional_modes)
    for mode in provisional_modes:
        mode["score_fraction_of_strongest"] = mode["selection_score"] / strongest_score if strongest_score > 0 else 0.0

    # Identify a central mode if present. 
    central_mode = next((mode for mode in provisional_modes if mode["zero_lag"]), None)

    # Prepare sets to record results prior to algorithm. 
    deleted_island_ids = set()
    rejected_provisional_mode_ids = set()
    central_gate = {"status": "absent"}
    candidate_modes = [dict(mode) for mode in provisional_modes]

    # If there is a central mode
    if central_mode is not None:
        central_member_ids = set(central_mode["islands"])
        central_ids = set(central_mode["compatible_islands"])

        # Identify every displaced mode that competes with a central mode. 
        conflict_pairs = {frozenset((conflict["island_1"], conflict["island_2"])) for conflict in conflicts}
        conflicting_modes = [
            mode for mode in provisional_modes
            if not mode["zero_lag"] and (
                bool(central_ids & set(mode["compatible_islands"]))
                or any(frozenset((central_id, displaced_id)) in conflict_pairs
                       for central_id in central_ids for displaced_id in mode["compatible_islands"])
            )
        ]
        
        # Identify strongest competing mode. 
        strongest_conflicting = max(conflicting_modes, key=lambda x: x["selection_score"]) if conflicting_modes else None

        # Reject the central mode if it is too weak relative to the strongest mode,
        # regardless of whether the two modes conflict.
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

        # Case where there is a conflicting mode strong enough that we cannot select the central mode. 
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
        # If the central mode survives, delete all conflicting islands (assume central)
        else:
            for conflict in conflicts:
                island_1, island_2 = conflict["island_1"], conflict["island_2"]
                if island_1 in central_ids and island_2 not in central_member_ids:
                    deleted_island_ids.add(island_2)
                elif island_2 in central_ids and island_1 not in central_member_ids:
                    deleted_island_ids.add(island_1)

            # Identify the islands that are not central or conflicting. 
            displaced_islands = [
                island for island in islands
                if island["island_id"] not in central_member_ids
                and island["island_id"] not in deleted_island_ids
            ]

            # Recover islands belonging to the central mode. 
            central_members = [island for island in islands if island["island_id"] in central_member_ids]

            # Rebuild non-deleted candidate modes. 
            candidate_modes = [summarize_correspondence_mode(central_members),
                               *correspondence_modes(displaced_islands)]

            # Record that the central mode survived. 
            central_gate = {
                "status": "retained", "reason": "central_mode_won_conflicts",
                "central_mode_id": central_mode["mode_id"],
                "central_score": central_mode["selection_score"],
                "central_lag_interval": (central_mode["lag_min"], central_mode["lag_max"]),
                "central_compatible_islands": central_mode["compatible_islands"],
                "strongest_conflicting_mode_id": strongest_conflicting["mode_id"] if strongest_conflicting else None,
                "strongest_conflicting_score": strongest_conflicting["selection_score"] if strongest_conflicting else None,
            }

    # identify islands and candidate modes that survived the central mode filtration. 
    surviving_islands = [island for island in islands if island["island_id"] not in deleted_island_ids]
    candidate_modes = number_modes(candidate_modes)

    # Find the strongest scoring mode. 
    strongest_score = max((mode["selection_score"] for mode in candidate_modes), default=0.0)

    # Calculate mode score fraction. 
    for mode in candidate_modes:
        mode["score_fraction_of_strongest"] = mode["selection_score"] / strongest_score if strongest_score > 0 else 0.0

    # Identify retained and ignored modes. 
    retained_candidates = [mode for mode in candidate_modes
                           if mode["score_fraction_of_strongest"] >= mode_score_fraction]
    rejected_mode_ids = {mode["mode_id"] for mode in candidate_modes if mode not in retained_candidates}

    # Create the final list of modes. 
    modes = []
    for mode_id, candidate in enumerate(retained_candidates, start=1):
        mode = dict(candidate)
        mode["candidate_mode_id"] = candidate["mode_id"]
        mode["mode_id"] = mode_id
        modes.append(mode)

    # Design a dictionary that specifies which islands belong to which modes. 
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

def plot_islands(result, title=None):
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
    island_mode_colors = {}
    for color_index, mode in enumerate(modes):
        color = mode_colors[color_index % len(mode_colors)]
        for island_id in mode["islands"]:
            island_mode_colors.setdefault(island_id, []).append(color)
    selected_island_ids = set(island_mode_colors)
    handles = []

    # Keep every detected ridge visible for displacement diagnostics.
    for island in islands:
        if island["island_id"] in selected_island_ids:
            continue
        ridge = island["ridge"]
        axis_x, axis_y = ridge[:, 1], ridge[:, 0]
        if len(axis_x) == 1:
            ax.plot(axis_x, axis_y, "o", color="black", ms=5)
            ax.plot(axis_x, axis_y, "o", color="#eeeeee", ms=3)
        else:
            ax.plot(axis_x, axis_y, color="black", lw=3)
            ax.plot(axis_x, axis_y, color="#eeeeee", lw=1.3)

    if len(selected_island_ids) < len(islands):
        handles.append(ax.plot([], [], color="#eeeeee", lw=1.3, label="unselected ridge")[0])

    # Nested outlines preserve every color when one broad ridge supports more
    # than one distinct, narrower lag hypothesis.
    for island_id, colors in island_mode_colors.items():
        mask = labels == island_id
        island = island_lookup[island_id]
        ridge = island["ridge"]
        axis_x, axis_y = ridge[:, 1], ridge[:, 0]
        widths = np.linspace(2 + 1.5 * (len(colors) - 1), 2, len(colors))
        ridge_widths = np.linspace(1.5 + 1.2 * (len(colors) - 1), 1.3, len(colors))

        if mask.any() and (~mask).any():
            for color, width in zip(colors, widths):
                ax.contour(mask.astype(float), levels=[0.5], colors=color,
                           linewidths=width, origin="lower")
        if len(axis_x) == 1:
            ax.plot(axis_x, axis_y, "o", color="black", ms=5 + 1.5 * len(colors))
            for color, marker_size in zip(colors, np.linspace(4 + len(colors), 3, len(colors))):
                ax.plot(axis_x, axis_y, "o", color=color, ms=marker_size)
        else:
            ax.plot(axis_x, axis_y, color="black", lw=max(ridge_widths) + 1.5)
            for color, width in zip(colors, ridge_widths):
                ax.plot(axis_x, axis_y, color=color, lw=width)

    for color_index, mode in enumerate(modes):
        color = mode_colors[color_index % len(mode_colors)]
        handles.append(ax.plot([], [], color=color, lw=2,
                               label=(f"mode {mode['mode_id']} "
                                      f"(lag {mode['lag_min']:+.1f} to {mode['lag_max']:+.1f})"))[0])

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


def identify_islands(signal_a, signal_b, seed_threshold_fraction=0.4, growth_threshold_fraction=0.39, max_weak_distance=2, gap_tolerance=1, shape_balance_power=1.05, ridge_fraction=0.9, split_prominence_fraction=0.25, split_min_peak_fraction=0.5, split_min_peak_distance=8, split_smoothing=4.0, split_min_seed_area=5, displacement_fold_threshold=2.0, mode_score_fraction=0.33, title=None, plot=True):
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
        split_prominence_fraction=split_prominence_fraction,
        split_min_peak_fraction=split_min_peak_fraction,
        split_min_peak_distance=split_min_peak_distance,
        split_smoothing=split_smoothing, split_min_seed_area=split_min_seed_area)
    conflicts = island_conflicts(labels, islands)
    selection = select_correspondence_modes(
        islands, conflicts, displacement_fold_threshold=displacement_fold_threshold,
        mode_score_fraction=mode_score_fraction)

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
        result["figure"], result["axes"] = plot_islands(result, title=title)

    return result
