# Validation

The baseline is `reference_utils.py`, copied without changes from the empty-lag-corrected utilities. Original project and Downloads files were not edited.

## Algorithm equivalence

`python validate_equivalence.py --cases 50` passed exact comparisons of complete outputs for 50 synthetic 300-position comparisons, containing 332 islands and 131 retained modes. Compared fields include all masks, labels, ridge coordinates, support and selection scores, candidate lag lists, compatible memberships, conflict diagnostics, and central-mode decisions.

Additional checks covered fractional and unsupported lags, the previously failing empty-lag case, constant/zero signals, and 30 split masks with holes and unsupported rows.

In the local single-core trial, the baseline took 28.66 seconds and the optimized implementation took 7.00 seconds for the same 50 comparisons, approximately 4.10 times faster. These are synthetic timings from this machine, not predictions for TSCC or the biological dataset.

## Notebook and worker integration

All three copied notebooks passed notebook-schema and Python-source validation.

The pilot notebook was executed end to end on a synthetic comparison with 32 windows, selecting 17 evenly distributed windows and stopping afterward. Its optional baseline comparison passed. No scatter or full-run analysis was launched.

The same 17-window sample produced exactly matching mode tables and selection summaries with one worker and eight worker processes. Signal archives retained the correct original window indices. An injected coverage error saved its window record and error log before stopping, with the run marked unfinished.

The real input archives are available only on TSCC and were not analyzed locally. The default notebook pilot is 256 windows and requests up to eight allocated CPUs.
