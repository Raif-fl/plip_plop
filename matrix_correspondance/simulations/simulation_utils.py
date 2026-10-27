import numpy as np

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