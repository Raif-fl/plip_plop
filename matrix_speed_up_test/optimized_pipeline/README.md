# Optimized correspondence pipeline

This is a separate copy of the pipeline. Existing project and Downloads files are unchanged. Keep this folder together when transferring it to TSCC.

## Start with the pilot

Open `mode_landscape.ipynb`, check the paths in Settings, and run top to bottom. Its defaults are:

- The smaller available PUM1/PUM2 or FUS/TAF15 comparison, chosen by window count.
- An optional eight-window comparison with the corrected baseline.
- A 256-window pilot sampled evenly across the selected comparison.
- Up to eight worker processes, limited by CPU affinity and `SLURM_CPUS_PER_TASK` when present.
- A full-run time projection followed by a stop. No remaining windows, scatter plots, or clustering run automatically.

Set `N_WORKERS = 1` to measure the algorithm alone, or `N_WORKERS = 8` to include process parallelism. Request eight CPUs for the Jupyter allocation when using eight workers. Native numerical-library threads are limited to one per process to avoid multiplying thread counts.

Set `BASELINE_CHECK_WINDOWS = 0` to skip the optional paired comparison. When enabled, the same windows must produce exactly matching mode tables and selection summaries. Its runtime is excluded from the pilot estimate.

The pilot's input arrays are loaded from the existing preparation outputs; there is no need to prepare the data again. Each run writes a new folder under `pilot_results`. The wall-clock estimates include actual worker overhead. Startup-adjusted and direct projections are both shown; neither is a guaranteed runtime for the unsampled windows.

## Files

| File | Purpose |
| --- | --- |
| `utils.py` | Optimized island/mode implementation, including the empty-lag fix. |
| `reference_utils.py` | Frozen corrected baseline for comparisons. |
| `island_identification.ipynb` | Original example workflow using this folder's optimized utilities. |
| `correspondence_lines_prototype.ipynb` | Original correspondence-line workflow and plotting. |
| `mode_landscape.ipynb` | Pilot-only batch analysis and time estimates. |
| `pipeline_runner.py` | Importable serial/process worker implementation. |
| `mode_line_metrics.py` | The existing quantile-line calculations shared by the runner. |
| `prepare_inputs.py` | Unchanged 300-nt input preparation script. |
| `prepare_initial_30.sbatch` | Preparation job pointing at this folder, with separate output defaults. |
| `validate_equivalence.py` | Reproducible comparisons of complete algorithm outputs and timings. |

Notebook execution counts and outputs were cleared in the copies. Plotting code was retained. The duplicated island notebook also uses the full window for the abundance reference and rejects old 600-position signal arrays.

Run notebooks from this folder. In the island notebook, edit `UTILS_PATH` if the kernel's working directory differs. Its local `pickles/` outputs can feed the correspondence-line notebook as before.

## Optimizations

1. Score the one or two relevant diagonals directly instead of scanning a full matrix for each island/lag.
2. Reuse bounds and cache island/lag summaries across candidate modes within each window.
3. Retain lightweight candidate scores; construct full masks and conflict diagnostics only for winning lags.
4. Evaluate split candidates within the island's bounding box, count seed support cumulatively, and construct child masks only for the winning cut.

Every original candidate lag is still evaluated. Thresholds, score definitions, floating-point summation order, tie-breakers, recursion, central-mode decisions, and line-metric definitions are preserved. No coarse lag search, GPU method, or parameter simplification was introduced.

Window-level process parallelism is optional. Only small per-window coverage arrays are sent to workers; full matrices and masks stay in the worker. Results are sorted by original window and mode ID before saving. Each process evaluates the same serial algorithm.

## Outputs and recovery

The pilot writes `mode_metrics.tsv`, `window_timings.tsv`, `pilot_windows.tsv`, `enrichment_signals.npz`, `run_report.json`, and `pilot_manifest.json`. The optional baseline check has separate reference and optimized subfolders.

The signal archive contains only windows with calculated signals. Its `window_idx` array maps archive rows to original input indices; archive row numbers are not generally original window indices.

Unexpected errors save completed records and an `errors.log` before stopping. Tables checkpoint every 100 windows or 30 seconds. Re-running the pilot creates a new directory; automatic resume is not implemented. The original landscape notebook and its previous checkpoints are untouched.

## Preparing inputs separately

Preparation behavior is unchanged. To use the copied Slurm script, submit it from this folder so `SLURM_SUBMIT_DIR` points here, or explicitly export `PIPELINE_DIR` to this folder. Ensure `logs/` exists before submitting.

The copied script defaults to `processed_inputs_optimized` rather than overwriting the original preparation output. If you use that output, update the notebook's `INPUT_DIR` accordingly. The comparison table remains a read-only input shared with the original workflow.

## Reproduce the regression checks

From this folder, run:

```bash
python validate_equivalence.py --cases 50
```

This compares complete result dictionaries, including masks, labels, ridges, scores, lag choices, central gates, and memberships. It includes the empty-lag case, fractional lags, tie cases, and split masks with holes. The benchmark reports single-core algorithm time on synthetic windows, not a biological-data runtime prediction.
