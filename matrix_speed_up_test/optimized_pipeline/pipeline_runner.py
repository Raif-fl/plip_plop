"""Run independent windows with bounded process parallelism."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from mode_line_metrics import MODE_EXTRA_COLUMNS, TIMING_COLUMNS, mode_metric_rows

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

_state = {}
_thread_limiter = None


def load_utils(path):
    spec = importlib.util.spec_from_file_location("isolated_pipeline_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_count(requested, n_windows):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation:
        available = min(available, int(allocation))
    if requested < 1 or requested > 8:
        raise ValueError("n_workers must be between 1 and 8.")
    return min(requested, available, n_windows)


@contextmanager
def single_thread_environment():
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(dict.fromkeys(keys, "1"))
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _initialize_worker(utils_path, library_sizes, configuration):
    global _state, _thread_limiter
    try:
        from threadpoolctl import threadpool_limits
        _thread_limiter = threadpool_limits(limits=1)
    except ImportError:
        _thread_limiter = None
    utils = load_utils(utils_path)
    _state = dict(utils=utils, library_sizes=library_sizes, configuration=configuration,
                  pseudocounts={key: utils.calculate_rpm_pseudocount(value) for key, value in library_sizes.items()})


def _analyze_window(payload):
    index, coverage, metadata = payload
    utils, configuration = _state["utils"], _state["configuration"]
    sizes, pseudocounts = _state["library_sizes"], _state["pseudocounts"]
    started = perf_counter()
    timing = {**metadata, "status": "ok", "n_modes": 0, "n_fitted_modes": 0, "n_islands": 0,
              "n_provisional_modes": 0, "evaluated_lags": 0, "central_gate": "", "central_gate_reason": "",
              "enrichment_seconds": 0.0, "islands_seconds": 0.0, "line_metrics_seconds": 0.0, "error": ""}
    stage, stage_started = "enrichment_seconds", perf_counter()
    signal_a = signal_b = None
    try:
        rpm = {}
        for group, array in coverage.items():
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"Invalid coverage in {group}, window {index}.")
            rpm[group] = array * (1e6 / sizes[group][:, None])
        mean_ab = (rpm["ip_A"].mean() + rpm["ip_B"].mean()) / 2
        signals = []
        for side in ("A", "B"):
            scores = utils.abundance_weighted_enrichment(rpm[f"ip_{side}"], rpm[f"in_{side}"][0],
                pseudocounts[f"ip_{side}"], pseudocounts[f"in_{side}"], mean_ab,
                p=configuration["abundance_p"], smoothing_sigma=configuration["smoothing_sigma"])
            signal = scores.astype(np.float32).mean(axis=0)
            if signal.shape != (300,) or not np.isfinite(signal).all():
                raise ValueError(f"Invalid enrichment in {side}, window {index}.")
            signals.append(signal)
        signal_a, signal_b = signals
        timing[stage] = perf_counter() - stage_started

        stage, stage_started = "islands_seconds", perf_counter()
        result = utils.identify_islands(signal_a, signal_b, plot=False, **configuration["island_parameters"])
        timing[stage] = perf_counter() - stage_started
        selection = result["selection"]
        timing.update(n_modes=len(result["modes"]), n_islands=len(result["islands"]),
            n_provisional_modes=len(selection["provisional_modes"]),
            evaluated_lags=sum(len(mode["evaluated_lags"]) for mode in selection["provisional_modes"]),
            central_gate=selection["central_gate"]["status"], central_gate_reason=selection["central_gate"].get("reason", ""))

        stage, stage_started = "line_metrics_seconds", perf_counter()
        rows = mode_metric_rows(result, metadata, configuration["quantile_parameters"], configuration["angle_quantiles"])
        timing[stage] = perf_counter() - stage_started
        timing["n_fitted_modes"] = sum(row["metric_status"] == "ok" for row in rows)
        timing["status"] = "no_modes" if not rows else ("ok" if timing["n_fitted_modes"] == len(rows) else "degenerate_modes")
        error_traceback = ""
    except Exception as exc:
        timing[stage] = perf_counter() - stage_started
        timing.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows, error_traceback = [], traceback.format_exc()
    timing["total_seconds"] = perf_counter() - started
    return index, rows, timing, signal_a, signal_b, error_traceback


def write_tsv(table, path):
    temporary = Path(str(path) + ".tmp")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def run_windows(raw_arrays, library_sizes, metadata, indices, configuration, run_dir, n_workers=1, utils_path=None):
    """Return small result tables and save completed windows at checkpoints."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    utils_path = Path(utils_path or Path(__file__).with_name("utils.py")).resolve()
    indices = [int(index) for index in indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Provide a nonempty sequence of unique window indices.")
    if min(indices) < 0 or max(indices) >= len(metadata):
        raise ValueError("Window indices are outside the loaded comparison.")
    n_workers = worker_count(n_workers, len(indices))
    records = metadata.set_index("window_idx", drop=False).to_dict("index")
    mode_rows, timing_rows, signals = [], [], {}
    started = last_checkpoint = perf_counter()
    first_result_seconds = None
    n_errors = 0
    finished = False
    progress = tqdm(total=len(indices), desc=f"Windows ({n_workers} workers)", unit="window") if tqdm else None

    def payload(index):
        return index, {key: value[index] for key, value in raw_arrays.items()}, records[index]

    def checkpoint():
        modes = pd.DataFrame(mode_rows, columns=list(metadata.columns) + MODE_EXTRA_COLUMNS)
        windows = pd.DataFrame(timing_rows, columns=list(metadata.columns) + TIMING_COLUMNS)
        if not modes.empty:
            modes = modes.sort_values(["window_idx", "mode_id"]).reset_index(drop=True)
        if not windows.empty:
            windows = windows.sort_values("window_idx").reset_index(drop=True)
        write_tsv(modes, run_dir / "mode_metrics.tsv")
        write_tsv(windows, run_dir / "window_timings.tsv")
        return modes, windows

    def consume(result):
        nonlocal first_result_seconds, n_errors, last_checkpoint
        index, rows, timing, signal_a, signal_b, error_traceback = result
        if first_result_seconds is None:
            first_result_seconds = perf_counter() - started
        mode_rows.extend(rows)
        timing_rows.append(timing)
        if signal_a is not None:
            signals[index] = (signal_a, signal_b)
        if error_traceback:
            n_errors += 1
            with (run_dir / "errors.log").open("a") as handle:
                handle.write(f"\nWindow {index}\n{error_traceback}")
        if progress:
            progress.update(1)
            progress.set_postfix(modes=len(mode_rows), errors=n_errors, refresh=False)
        if len(timing_rows) % configuration.get("checkpoint_windows", 100) == 0 or perf_counter() - last_checkpoint >= configuration.get("checkpoint_seconds", 30):
            checkpoint()
            last_checkpoint = perf_counter()
            if not progress:
                print(f"{len(timing_rows)}/{len(indices)} windows; {perf_counter() - started:.1f} s elapsed")
        if error_traceback and configuration.get("stop_on_error", True):
            raise RuntimeError(f"Window {index}: {timing['error']}; see {run_dir / 'errors.log'}")

    try:
        with single_thread_environment():
            if n_workers == 1:
                _initialize_worker(str(utils_path), library_sizes, configuration)
                for index in indices:
                    consume(_analyze_window(payload(index)))
            else:
                # Keep at most two pending windows per worker.
                executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker, initargs=(str(utils_path), library_sizes, configuration))
                pending = set()
                iterator = iter(indices)
                try:
                    for _ in range(min(len(indices), 2 * n_workers)):
                        pending.add(executor.submit(_analyze_window, payload(next(iterator))))
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            consume(future.result())
                            next_index = next(iterator, None)
                            if next_index is not None:
                                pending.add(executor.submit(_analyze_window, payload(next_index)))
                finally:
                    for future in pending:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
        finished = True
    finally:
        if n_workers == 1 and _thread_limiter is not None:
            _thread_limiter.restore_original_limits()
        if progress:
            progress.close()
        modes, windows = checkpoint()
        available = sorted(signals)
        np.savez_compressed(run_dir / "enrichment_signals.npz", window_idx=np.asarray(available, dtype=int),
            signal_A=np.stack([signals[index][0] for index in available]) if available else np.empty((0, 300), np.float32),
            signal_B=np.stack([signals[index][1] for index in available]) if available else np.empty((0, 300), np.float32))
        report = dict(finished=finished, n_requested=len(indices), n_processed=len(windows), n_errors=n_errors,
            n_workers=n_workers, n_modes=len(modes), wall_seconds=perf_counter() - started,
            first_result_seconds=first_result_seconds, utils_path=str(utils_path),
            utils_sha256=hashlib.sha256(utils_path.read_bytes()).hexdigest(), configuration=configuration,
            updated_utc=datetime.now(timezone.utc).isoformat())
        (run_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    return modes, windows, report
