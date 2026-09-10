# Load necessary packages. 
import os
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib.pyplot as plt
import subprocess
from scipy.ndimage import gaussian_filter1d

def get_track_paths(row, strand, track_type, side):
    paths = []

    if track_type == "IP":
        prefix = f"{strand}_"
        suffix = f"_{side}"

        for column in row.index:
            if column.startswith(prefix) and column.endswith(suffix) and "_IN" not in column and column not in [f"window_{side}"]:
                path = row[column]

                if pd.notna(path) and str(path).strip():
                    paths.append(str(path))

    elif track_type == "IN":
        prefix = f"{strand}_IN"
        suffix = f"_{side}"

        for column in row.index:
            if column.startswith(prefix) and column.endswith(suffix):
                path = row[column]

                if pd.notna(path) and str(path).strip():
                    paths.append(str(path))

    return(paths)

def load_regions(window_files, buffer=100, block_size=300):
    # Read and combine Skipper reproducible-window tables.
    regions = pd.concat([pd.read_csv(f, sep="\t") for f in window_files], ignore_index=True)

    # Add buffer around each region.
    regions["start_buffered"] = (regions["start"] - buffer).clip(lower=0)
    regions["end_buffered"] = regions["end"] + buffer

    # Sort so overlapping windows can be merged separately by strand.
    regions = regions.sort_values(["chr", "strand", "start_buffered", "end_buffered"])

    merged = []

    for (chrom, strand), group in regions.groupby(["chr", "strand"], sort=False):
        current_start = None
        current_end = None

        for row in group.itertuples(index=False):
            if current_start is None:
                current_start = row.start_buffered
                current_end = row.end_buffered

            elif row.start_buffered < current_end:
                current_end = max(current_end, row.end_buffered)

            else:
                merged.append({
                    "chr": chrom,
                    "strand": strand,
                    "start_buffered": current_start,
                    "end_buffered": current_end
                })

                current_start = row.start_buffered
                current_end = row.end_buffered

        merged.append({
            "chr": chrom,
            "strand": strand,
            "start_buffered": current_start,
            "end_buffered": current_end
        })

    blocks = []

    for region_id, region in enumerate(merged):
        chrom = region["chr"]
        strand = region["strand"]
        start = region["start_buffered"]
        end = region["end_buffered"]

        length = end - start

        # Expand short regions to the target block size.
        if length < block_size:
            extra = block_size - length
            start = max(0, start - extra // 2)
            end = start + block_size

        block_start = start
        block_number = 0

        while block_start + block_size <= end:
            blocks.append({
                "chr": chrom,
                "strand": strand,
                "start_buffered": block_start,
                "end_buffered": block_start + block_size,
                "region_id": region_id,
                "block_number": block_number
            })

            block_start += block_size
            block_number += 1

        # Shift the final block backward to preserve block length.
        if block_start < end:
            blocks.append({
                "chr": chrom,
                "strand": strand,
                "start_buffered": end - block_size,
                "end_buffered": end,
                "region_id": region_id,
                "block_number": block_number
            })

    return(pd.DataFrame(blocks))

def extract_bigwig_signal(
    plus_bigwigs,
    minus_bigwigs,
    plus_library_sizes,
    minus_library_sizes,
    regions
):

    plus_bws = [pyBigWig.open(path) for path in plus_bigwigs]
    minus_bws = [pyBigWig.open(path) for path in minus_bigwigs]

    extracted = []

    try:
        for row in regions.itertuples(index=False):

            if row.strand == "+":
                bws = plus_bws
                library_sizes = plus_library_sizes
            elif row.strand == "-":
                bws = minus_bws
                library_sizes = minus_library_sizes
            else:
                continue

            if not all(row.chr in bw.chroms() for bw in bws):
                continue

            chrom_size = min(bw.chroms(row.chr) for bw in bws)

            start = max(0, row.start_buffered)
            end = min(row.end_buffered, chrom_size)

            raw_replicate_values = []
            scaled_replicate_values = []

            for bw, library_size in zip(bws, library_sizes):
                values = bw.values(
                    row.chr,
                    start,
                    end,
                    numpy=True
                )

                # Convert missing bigWig values to zero.
                values = np.nan_to_num(values, nan=0.0)

                # Keep the original unscaled coverage.
                raw_replicate_values.append(values)

                # Scale each replicate independently to reads per million.
                scaled_values = values * (1e6 / library_size)
                scaled_replicate_values.append(scaled_values)

            # Average biological replicates.
            raw_values = np.mean(raw_replicate_values, axis=0)
            scaled_values = np.mean(scaled_replicate_values, axis=0)

            # Orient minus-strand windows in 5' -> 3' direction.
            if row.strand == "-":
                raw_values = raw_values[::-1]
                scaled_values = scaled_values[::-1]

            extracted.append({
                "chr": row.chr,
                "start": start,
                "end": end,
                "strand": row.strand,
                "region_id": row.region_id,
                "block_number": row.block_number,
                "raw_replicates": np.stack(raw_replicate_values),
                "raw_signal": raw_values,
                "signal": scaled_values
            })

    finally:
        for bw in plus_bws + minus_bws:
            bw.close()

    return(extracted)

def plot_basic_window(window_data, window_id, experiment_A, experiment_B):
    window = window_data[window_id]
    x = np.arange(len(window["ip_A"]))

    ymax = max(
        window["ip_A"].max(),
        window["ip_B"].max(),
        window["in_A"].max(),
        window["in_B"].max()
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot raw IP signal.
    axes[0].plot(x, window["ip_A"], label=experiment_A)
    axes[0].plot(x, window["ip_B"], label=experiment_B)
    axes[0].set_ylabel("IP signal (RPM)")
    axes[0].set_ylim(0, ymax)
    axes[0].legend()

    # Plot raw input signal.
    axes[1].plot(x, window["in_A"], label=experiment_A)
    axes[1].plot(x, window["in_B"], label=experiment_B)
    axes[1].set_ylabel("IN signal (RPM)")
    axes[1].set_ylim(0, ymax)
    axes[1].legend()

    fig.suptitle(
        f"Window {window_id}: "
        f"{window['chr']}:{window['start']}-{window['end']} "
        f"({window['strand']})"
    )

    plt.tight_layout()
    plt.show()

def plot_dumb_IN_window(window_data, window_id, experiment_A, experiment_B):
    window = window_data[window_id]

    x = np.arange(len(window["ip_A"]))

    fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True)

    # Plot current input-corrected signal.
    axes.plot(x, window["corrected_A"], label=experiment_A)
    axes.plot(x, window["corrected_B"], label=experiment_B)
    axes.axhline(0, linewidth=1)
    axes.set_ylabel("log1p(IP) - log1p(IN)")
    axes.set_xlabel("Position within 300-nt window")
    axes.legend()

    fig.suptitle(
        f"Window {window_id}: "
        f"{window['chr']}:{window['start']}-{window['end']} "
        f"({window['strand']})"
    )

    plt.tight_layout()
    plt.show()

def get_mapped_reads(bam_file):
    result = subprocess.run(["samtools", "idxstats", bam_file], capture_output=True, text=True, check=True)

    mapped_reads = sum(
        int(line.split("\t")[2])
        for line in result.stdout.strip().split("\n")
        if line
    )

    return(mapped_reads)

def calculate_rpm_pseudocount(library_sizes):
    library_sizes = np.atleast_1d(library_sizes).astype(float)

    return(np.mean(1e6 / library_sizes))

def abundance_weighted_enrichment(ip, in_signal, ip_pseudocount, in_pseudocount, p=1):
    mean_window_signal = (ip + in_signal).mean()
    k = p * mean_window_signal

    log_enrichment = np.log(
        (ip + ip_pseudocount) /
        (in_signal + in_pseudocount)
    )

    abundance_weight = (ip + in_signal) / (ip + in_signal + k)

    return(log_enrichment * abundance_weight)

def plot_abundance_IN_window(window_data, window_id, experiment_A, experiment_B):
    window = window_data[window_id]

    x = np.arange(len(window["ip_A"]))

    fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True)

    axes.plot(x, window["abundance_corrected_A"], label=experiment_A)

    axes.plot(x, window["abundance_corrected_B"], label=experiment_B)

    axes.axhline(0, linewidth=1)
    axes.set_ylabel("Abundance-weighted enrichment")
    axes.set_xlabel("Position within 300-nt window")
    axes.legend()

    fig.suptitle(
        f"Window {window_id}: "
        f"{window['chr']}:{window['start']}-{window['end']} "
        f"({window['strand']})"
    )

    plt.tight_layout()
    plt.show()

def smooth_signal(signal, sigma=1):
    signal = np.asarray(signal, dtype=float)

    return(gaussian_filter1d(signal, sigma=sigma))