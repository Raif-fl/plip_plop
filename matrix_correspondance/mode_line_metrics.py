"""Quantile-line metrics shared by the batch and pilot runners."""

import numpy as np

def surviving_mode_regions(data):
    """Pool the selected member masks for each retained mode."""
    matrix = data["matrix"]
    modes = []
    for source_mode in data["modes"]:
        member_masks = {}
        for island_id in source_mode["islands"]:
            mask = np.asarray(source_mode["constrained_regions"][island_id], dtype=bool)
            if mask.shape != matrix.shape or not mask.any():
                raise ValueError(f"Invalid mask for island {island_id}.")
            member_masks[int(island_id)] = mask
        if not member_masks:
            raise ValueError(f'Mode {source_mode["mode_id"]} has no supported pixels.')
        mode_mask = np.logical_or.reduce(list(member_masks.values()))
        rows, columns = np.where(mode_mask)
        modes.append(dict(mode_id=int(source_mode["mode_id"]), island_ids=tuple(member_masks),
                          mask=mode_mask, x_min=int(columns.min()), x_max=int(columns.max()),
                          y_min=int(rows.min()), y_max=int(rows.max()), source=source_mode))
    return modes


def mode_crop(array, mode):
    return array[mode["y_min"]:mode["y_max"] + 1, mode["x_min"]:mode["x_max"] + 1]


def weighted_quantile_positions(weights, quantiles):
    """Convert cumulative mass to fractional positions."""
    weights = np.clip(np.asarray(weights, dtype=float), 0, None)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    cumulative = np.r_[0.0, np.cumsum(weights) / weights.sum()]
    positions = np.linspace(0, len(weights) - 1, len(weights) + 1)
    return np.interp(quantiles, cumulative, positions)


def smooth_marginal(weights, sigma):
    if sigma <= 0:
        return weights
    radius = max(1, int(np.ceil(4 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(weights, radius, mode="edge"), kernel, mode="valid")


def prepare_marginal(weights, signal_influence, smoothing_sigma, support):
    """Blend the observed marginal with uniform supported mass."""
    support = np.asarray(support, dtype=bool)
    weights = smooth_marginal(np.asarray(weights, dtype=float), smoothing_sigma)
    weights = np.where(support, weights, 0)
    uniform = support / support.sum()
    observed = uniform if weights.sum() == 0 else weights / weights.sum()
    return signal_influence * observed + (1 - signal_influence) * uniform


def project_points_to_mask(x, y, mask):
    """Move unsupported points to their nearest valid pixel."""
    valid_rows, valid_columns = np.where(mask)
    if len(valid_rows) == 0:
        raise ValueError("Cannot project onto an empty mode mask.")
    projected_x = np.asarray(x, dtype=float).copy()
    projected_y = np.asarray(y, dtype=float).copy()
    rounded_x = np.rint(projected_x).astype(int)
    rounded_y = np.rint(projected_y).astype(int)
    in_bounds = ((rounded_x >= 0) & (rounded_x < mask.shape[1])
                 & (rounded_y >= 0) & (rounded_y < mask.shape[0]))
    valid = np.zeros(len(projected_x), dtype=bool)
    valid[in_bounds] = mask[rounded_y[in_bounds], rounded_x[in_bounds]]
    for index in np.where(~valid)[0]:
        distances = (valid_columns - projected_x[index]) ** 2 + (valid_rows - projected_y[index]) ** 2
        nearest = int(np.argmin(distances))
        projected_x[index], projected_y[index] = valid_columns[nearest], valid_rows[nearest]
    return projected_x, projected_y


def quantile_matching_line(matrix, mode, n_points=128, signal_influence=1.0, signal_power=1.0, smoothing_sigma=0.0):
    if not 0 <= signal_influence <= 1 or signal_power <= 0 or n_points < 3:
        raise ValueError("Check signal_influence, signal_power, and n_points.")
    crop = mode_crop(matrix, mode)
    mask_crop = mode_crop(mode["mask"], mode)
    positive_crop = np.where(mask_crop, np.clip(crop, 0, None), 0)
    peak = positive_crop.max()
    weighted_crop = (positive_crop / peak) ** signal_power if peak > 0 else positive_crop
    b_marginal = prepare_marginal(weighted_crop.sum(axis=0), signal_influence, smoothing_sigma, mask_crop.any(axis=0))
    a_marginal = prepare_marginal(weighted_crop.sum(axis=1), signal_influence, smoothing_sigma, mask_crop.any(axis=1))
    quantiles = np.linspace(0, 1, n_points)
    x = weighted_quantile_positions(b_marginal, quantiles) + mode["x_min"]
    y = weighted_quantile_positions(a_marginal, quantiles) + mode["y_min"]
    return project_points_to_mask(x, y, mode["mask"])


def quantile_angle_metrics(quantile_x, quantile_y, mode, lower_quantile=0.20, upper_quantile=0.80):
    """Measure the prototype's angle and median displacement."""
    if not 0 < lower_quantile < 0.5 < upper_quantile < 1:
        raise ValueError("Use lower < 0.5 < upper, strictly between 0 and 1.")
    quantile_grid = np.linspace(0, 1, len(quantile_x))
    requested = np.array([lower_quantile, 0.5, upper_quantile])
    landmark_b, landmark_a = project_points_to_mask(
        np.interp(requested, quantile_grid, quantile_x),
        np.interp(requested, quantile_grid, quantile_y), mode["mask"]
    )
    b_low, b_median, b_high = landmark_b
    a_low, a_median, a_high = landmark_a
    delta_b, delta_a = b_high - b_low, a_high - a_low
    box_delta_b = mode["x_max"] - mode["x_min"]
    box_delta_a = mode["y_max"] - mode["y_min"]
    if np.hypot(delta_b, delta_a) == 0 or box_delta_b == 0 or box_delta_a == 0:
        return None

    signal_slope = delta_a / delta_b if delta_b != 0 else np.inf
    box_slope = box_delta_a / box_delta_b
    signal_angle = np.degrees(np.arctan2(delta_a, delta_b))
    box_angle = np.degrees(np.arctan2(box_delta_a, box_delta_b))
    median_b_offset = b_median - mode["x_min"]
    median_a_offset = a_median - mode["y_min"]
    displacement_nt = (box_delta_b * median_a_offset - box_delta_a * median_b_offset) / np.hypot(box_delta_b, box_delta_a)
    displacement_normalized = (median_a_offset / box_delta_a - median_b_offset / box_delta_b) / np.sqrt(2)
    return dict(signal_angle_deg=float(signal_angle), box_angle_deg=float(box_angle),
                angle_offset_deg=float(signal_angle - box_angle),
                signal_slope=float(signal_slope), box_slope=float(box_slope),
                relative_slope=float(signal_slope / box_slope),
                median_b=float(b_median), median_a=float(a_median),
                median_lag_b_minus_a=float(b_median - a_median),
                median_displacement_nt=float(displacement_nt),
                median_displacement_normalized=float(displacement_normalized))

def clip_infinite_line_to_mask(center, direction, mask):
    """Return the pieces of an infinite line lying inside True mask pixels."""
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError('Cannot clip a line with zero direction.')
    direction = direction / norm

    intervals = []
    rows, columns = np.where(mask)
    tolerance = 1e-10

    # Treat each valid nucleotide as a one-by-one square centered on its index.
    for row, column in zip(rows, columns):
        lower, upper = -np.inf, np.inf
        for coordinate, velocity, minimum, maximum in (
            (center[0], direction[0], column - 0.5, column + 0.5),
            (center[1], direction[1], row - 0.5, row + 0.5),
        ):
            if abs(velocity) <= tolerance:
                if not minimum <= coordinate <= maximum:
                    lower, upper = 1, 0
                    break
            else:
                first = (minimum - coordinate) / velocity
                second = (maximum - coordinate) / velocity
                lower = max(lower, min(first, second))
                upper = min(upper, max(first, second))

        if lower <= upper + tolerance:
            intervals.append((lower, upper))

    if not intervals:
        return np.array([np.nan]), np.array([np.nan])

    intervals.sort()
    merged = [list(intervals[0])]
    for lower, upper in intervals[1:]:
        if lower <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], upper)
        else:
            merged.append([lower, upper])

    x, y = [], []
    for lower, upper in merged:
        start = center + lower * direction
        end = center + upper * direction
        if x:
            x.append(np.nan)
            y.append(np.nan)
        x.extend([start[0], end[0]])
        y.extend([start[1], end[1]])

    return np.asarray(x), np.asarray(y)


def mask_polyline(x, y, mask):
    """Exactly clip every quantile-curve segment to the True mask pixels."""
    valid_rows, valid_columns = np.where(mask)
    output_x, output_y = [], []
    last_end = None
    tolerance = 1e-10

    for index in range(len(x) - 1):
        start = np.array([x[index], y[index]], dtype=float)
        end = np.array([x[index + 1], y[index + 1]], dtype=float)
        direction = end - start

        # Only pixels whose squares overlap this segment's bounding box can matter.
        candidates = (
            (valid_columns + 0.5 >= min(start[0], end[0]))
            & (valid_columns - 0.5 <= max(start[0], end[0]))
            & (valid_rows + 0.5 >= min(start[1], end[1]))
            & (valid_rows - 0.5 <= max(start[1], end[1]))
        )
        intervals = []

        for row, column in zip(valid_rows[candidates], valid_columns[candidates]):
            lower, upper = 0.0, 1.0
            for coordinate, velocity, minimum, maximum in (
                (start[0], direction[0], column - 0.5, column + 0.5),
                (start[1], direction[1], row - 0.5, row + 0.5),
            ):
                if abs(velocity) <= tolerance:
                    if not minimum <= coordinate <= maximum:
                        lower, upper = 1, 0
                        break
                else:
                    first = (minimum - coordinate) / velocity
                    second = (maximum - coordinate) / velocity
                    lower = max(lower, min(first, second))
                    upper = min(upper, max(first, second))

            if lower <= upper + tolerance:
                intervals.append((lower, upper))

        intervals.sort()
        merged = []
        for lower, upper in intervals:
            if merged and lower <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], upper)
            else:
                merged.append([lower, upper])

        for lower, upper in merged:
            clipped_start = start + lower * direction
            clipped_end = start + upper * direction
            if last_end is None or not np.allclose(last_end, clipped_start):
                if output_x:
                    output_x.append(np.nan)
                    output_y.append(np.nan)
                output_x.append(clipped_start[0])
                output_y.append(clipped_start[1])
            output_x.append(clipped_end[0])
            output_y.append(clipped_end[1])
            last_end = clipped_end

        if not merged:
            last_end = None

    if not output_x:
        return np.array([np.nan]), np.array([np.nan])
    return np.asarray(output_x), np.asarray(output_y)


def quantile_plot_lines(matrix, mode, quantile_parameters, angle_quantiles=(0.20, 0.80)):
    """Build clipped quantile and angle lines using the metric definitions."""
    quantile_x, quantile_y = quantile_matching_line(matrix, mode, **quantile_parameters)
    metrics = quantile_angle_metrics(quantile_x, quantile_y, mode, *angle_quantiles)
    grid = np.linspace(0, 1, len(quantile_x))
    requested = [angle_quantiles[0], 0.5, angle_quantiles[1]]
    landmark_b, landmark_a = project_points_to_mask(
        np.interp(requested, grid, quantile_x), np.interp(requested, grid, quantile_y), mode["mask"])

    # The angle direction spans the outer quantiles and passes through the median.
    straight = (np.array([np.nan]), np.array([np.nan]))
    if metrics is not None:
        center = np.array([landmark_b[1], landmark_a[1]])
        direction = np.array([landmark_b[2] - landmark_b[0], landmark_a[2] - landmark_a[0]])
        straight = clip_infinite_line_to_mask(center, direction, mode["mask"])

    return {
        "quantile": (quantile_x, quantile_y),
        "visible_quantile": mask_polyline(quantile_x, quantile_y, mode["mask"]),
        "straight": straight, "metrics": metrics, "landmarks": (landmark_b, landmark_a),
        "box_diagonal": (np.array([mode["x_min"], mode["x_max"]]), np.array([mode["y_min"], mode["y_max"]])),
        "angle_quantiles": tuple(angle_quantiles),
    }


METRIC_COLUMNS = ["signal_angle_deg", "box_angle_deg", "angle_offset_deg", "signal_slope",
                  "box_slope", "relative_slope", "median_b", "median_a", "median_lag_b_minus_a",
                  "median_displacement_nt", "median_displacement_normalized"]
MODE_EXTRA_COLUMNS = ["mode_id", "mode_key", "island_ids", "n_islands", "mode_pixels",
                      "lag_min", "lag_max", "zero_lag", "mode_scoring_method",
                      "total_shape_adjusted_signal", "mode_total_support",
                      "angle_deg", "displacement_nt", "metric_status"] + METRIC_COLUMNS
TIMING_COLUMNS = ["status", "n_modes", "n_fitted_modes", "n_islands", "n_provisional_modes",
                  "evaluated_lags", "central_gate", "central_gate_reason", "enrichment_seconds",
                  "islands_seconds", "line_metrics_seconds", "total_seconds", "error"]


def mode_metric_rows(result, metadata_row, quantile_parameters, angle_quantiles):
    rows = []
    for mode in surviving_mode_regions(result):
        source = mode["source"]
        quantile_x, quantile_y = quantile_matching_line(result["matrix"], mode, **quantile_parameters)
        metrics = quantile_angle_metrics(quantile_x, quantile_y, mode, *angle_quantiles)
        row = {**metadata_row, "mode_id": mode["mode_id"],
               "mode_key": f'{metadata_row["window_key"]}:m{mode["mode_id"]}',
               "island_ids": ",".join(map(str, mode["island_ids"])),
               "n_islands": len(mode["island_ids"]), "mode_pixels": int(mode["mask"].sum()),
               "lag_min": float(source["lag_min"]),
               "lag_max": float(source["lag_max"]), "zero_lag": bool(source["zero_lag"]),
               "total_shape_adjusted_signal": float(source["selection_score"]),
               "mode_scoring_method": source.get("scoring_method", "best_lag"),
               "mode_total_support": float(source["total_support"] if source.get("scoring_method") == "interval_region"
                                           else source["compatible_total_support"]),
               "metric_status": "ok" if metrics is not None else "degenerate_line"}
        row.update(metrics if metrics is not None else dict.fromkeys(METRIC_COLUMNS, np.nan))
        row["angle_deg"] = row["angle_offset_deg"]
        row["displacement_nt"] = row["median_displacement_nt"]
        rows.append(row)
    return rows

