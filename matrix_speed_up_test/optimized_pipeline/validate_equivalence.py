"""Compare the optimized implementation with the corrected baseline."""

import argparse
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_utils(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_same(before, after, path="result"):
    if isinstance(before, dict):
        assert before.keys() == after.keys(), path
        for key in before:
            assert_same(before[key], after[key], f"{path}.{key}")
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after, err_msg=path)
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after), path
        for index, (left, right) in enumerate(zip(before, after)):
            assert_same(left, right, f"{path}[{index}]")
    elif isinstance(before, set):
        assert before == after, path
    elif isinstance(before, (float, np.floating)) and np.isnan(before):
        assert np.isnan(after), path
    else:
        assert before == after, (path, before, after)


PARAMETERS = dict(seed_threshold_fraction=0.4, growth_threshold_fraction=0.2,
    max_weak_distance=5, gap_tolerance=2, shape_balance_power=1.05, ridge_fraction=0.6,
    split_prominence_fraction=0.15, split_min_peak_fraction=0.5, split_min_peak_distance=8,
    split_min_seed_area=5, displacement_fold_threshold=2.0, mode_score_fraction=0.33)


def signal_pairs(count):
    rng = np.random.default_rng(1027)
    positions = np.arange(300)
    for index in range(count):
        if index == 0:
            yield np.zeros(300), np.zeros(300)
            continue
        if index == 1:
            yield np.ones(300), np.ones(300)
            continue
        signals = []
        shared_centers = rng.uniform(5, 295, rng.integers(1, 6))
        for side in range(2):
            signal = np.zeros(300)
            centers = shared_centers if index % 3 else rng.uniform(5, 295, len(shared_centers))
            for center in centers:
                width = rng.uniform(2, 35)
                amplitude = rng.uniform(0.2, 6)
                shifted = center + (rng.uniform(-20, 20) if side else 0)
                signal += amplitude * np.exp(-0.5 * ((positions - shifted) / width) ** 2)
            if index % 4 == 0:
                signal += gaussian_filter1d(rng.uniform(0, 1, 300), 2)
            if index % 7 == 0:
                signal = np.round(signal, 1)
            signals.append(signal)
        yield signals


def run_checks(cases=40):
    root = Path(__file__).resolve().parent
    reference = load_utils(root / "reference_utils.py", "reference_validation_utils")
    optimized = load_utils(root / "utils.py", "optimized_validation_utils")

    matrix = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    region = matrix > 0
    labels = region.astype(np.int32)
    ridge = reference.island_central_ridge(matrix, region)
    island = dict(island_id=1, ridge=ridge, lag_min=-0.5, lag_max=0.5)
    row_indices, column_indices = np.indices(matrix.shape)
    cell_lags = column_indices - row_indices
    for lag in [-3, -1, -0.5, 0, 1e-17, 0.25, 0.5, 1, 3]:
        assert_same(reference.constrain_island_to_lag(matrix, labels, island, cell_lags, lag, 1.05),
                    optimized.constrain_island_to_lag(matrix, labels, island, cell_lags, lag, 1.05), f"lag {lag}")
    for lower, upper in [(-0.5, 0.5), (0, 0)]:
        assert_same(reference.build_correspondence_mode([island], matrix, labels, 1.05, lower, upper),
                    optimized.build_correspondence_mode([island], matrix, labels, 1.05, lower, upper))

    reference_seconds = optimized_seconds = 0.0
    n_modes = n_islands = 0
    for index, (signal_a, signal_b) in enumerate(signal_pairs(cases)):
        started = perf_counter()
        before = reference.identify_islands(signal_a, signal_b, plot=False, **PARAMETERS)
        reference_seconds += perf_counter() - started
        started = perf_counter()
        after = optimized.identify_islands(signal_a, signal_b, plot=False, **PARAMETERS)
        optimized_seconds += perf_counter() - started
        assert_same(before, after, f"window {index}")
        n_modes += len(after["modes"])
        n_islands += len(after["islands"])

    # Compare valley selection on masks that contain holes and unsupported rows.
    rng = np.random.default_rng(703)
    for index in range(30):
        matrix = rng.integers(0, 8, (37, 51)).astype(np.float32)
        region = rng.random(matrix.shape) > 0.4
        region[10:12] = False
        seeds = (matrix > 4) & region
        for axis in ("A", "B"):
            assert_same(reference.island_axis_profile(matrix, region, axis), optimized.island_axis_profile(matrix, region, axis))
        parameters = dict(split_min_peak_distance=2, split_min_seed_area=2, split_prominence_fraction=0.05)
        assert_same(reference.best_island_split(matrix, region, seeds, **parameters),
                    optimized.best_island_split(matrix, region, seeds, **parameters), f"split {index}")

    report = dict(windows=cases, islands=n_islands, modes=n_modes,
                  reference_seconds=reference_seconds, optimized_seconds=optimized_seconds,
                  single_core_speedup=reference_seconds / optimized_seconds)
    print("Exact output equality passed:", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=40)
    run_checks(parser.parse_args().cases)
