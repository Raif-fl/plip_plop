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

def extract_bigwig_signal(plus_bigwigs, minus_bigwigs, regions):
    plus_bws = [pyBigWig.open(path) for path in plus_bigwigs]
    minus_bws = [pyBigWig.open(path) for path in minus_bigwigs]

    extracted = []

    try:
        for row in regions.itertuples(index=False):

            if row.strand == "+":
                bws = plus_bws
            elif row.strand == "-":
                bws = minus_bws
            else:
                continue

            if not all(row.chr in bw.chroms() for bw in bws):
                continue

            chrom_size = min(bw.chroms(row.chr) for bw in bws)
            start = max(0, row.start_buffered)
            end = min(row.end_buffered, chrom_size)

            replicate_signals = np.stack([
                np.nan_to_num(bw.values(row.chr, start, end, numpy=True), nan=0.0)
                for bw in bws
            ])

            # Orient minus-strand windows in 5' -> 3' direction.
            if row.strand == "-":
                replicate_signals = replicate_signals[:, ::-1]

            extracted.append({
                "chr": row.chr,
                "start": start,
                "end": end,
                "strand": row.strand,
                "region_id": row.region_id,
                "block_number": row.block_number,
                "replicate_signals": replicate_signals
            })

    finally:
        for bw in plus_bws + minus_bws:
            bw.close()

    return extracted

def get_mapped_reads(bam_file):
    result = subprocess.run(["samtools", "idxstats", bam_file], capture_output=True, text=True, check=True)

    mapped_reads = sum(
        int(line.split("\t")[2])
        for line in result.stdout.strip().split("\n")
        if line
    )

    return(mapped_reads)