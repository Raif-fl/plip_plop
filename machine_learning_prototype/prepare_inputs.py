#!/usr/bin/env python3

# Import necessary packages.

import argparse
import numpy as np
import pandas as pd
import pyBigWig


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

def save_pair_dataset(signal_A, signal_B, output_file, rbp_A, rbp_B, cell_type):
    if len(signal_A) != len(signal_B):
        raise ValueError("RBP A and RBP B contain different numbers of blocks.")

    # Confirm that the genomic blocks match exactly.
    for a, b in zip(signal_A, signal_B):
        if (a["chr"], a["start"], a["end"], a["strand"]) != (b["chr"], b["start"], b["end"], b["strand"]):
            raise ValueError("RBP A and RBP B blocks do not match.")

    signals = np.stack([
        np.stack([a["signal"], b["signal"]])
        for a, b in zip(signal_A, signal_B)
    ]).astype(np.float32)

    np.savez_compressed(
        output_file,
        signals=signals,
        chrom=np.array([x["chr"] for x in signal_A]),
        start=np.array([x["start"] for x in signal_A]),
        end=np.array([x["end"] for x in signal_A]),
        strand=np.array([x["strand"] for x in signal_A]),
        region_id=np.array([x["region_id"] for x in signal_A]),
        block_number=np.array([x["block_number"] for x in signal_A]),
        rbp_A=np.array(rbp_A),
        rbp_B=np.array(rbp_B),
        cell_type=np.array(cell_type)
    )


def infer_cell_type(experiment):
    if "HepG2" in experiment:
        return("HepG2")
    elif "K562" in experiment:
        return("K562")
    else:
        return("unknown")

def extract_bigwig_signal(plus_ip_bigwigs, minus_ip_bigwigs, plus_in_bigwigs, minus_in_bigwigs, regions):
    plus_ip_bws = [pyBigWig.open(f) for f in plus_ip_bigwigs]
    minus_ip_bws = [pyBigWig.open(f) for f in minus_ip_bigwigs]
    plus_in_bws = [pyBigWig.open(f) for f in plus_in_bigwigs]
    minus_in_bws = [pyBigWig.open(f) for f in minus_in_bigwigs]

    extracted = []

    for row in regions.itertuples(index=False):

        if row.strand == "+":
            ip_bws = plus_ip_bws
            in_bws = plus_in_bws
        elif row.strand == "-":
            ip_bws = minus_ip_bws
            in_bws = minus_in_bws
        else:
            continue

        all_bws = ip_bws + in_bws

        if not all(row.chr in bw.chroms() for bw in all_bws):
            continue

        chrom_size = min(bw.chroms(row.chr) for bw in all_bws)

        start = max(0, row.start_buffered)
        end = min(row.end_buffered, chrom_size)

        ip_replicate_values = []
        in_replicate_values = []

        for bw in ip_bws:
            values = bw.values(row.chr, start, end, numpy=True)
            values = np.nan_to_num(values, nan=0.0)
            ip_replicate_values.append(values)

        for bw in in_bws:
            values = bw.values(row.chr, start, end, numpy=True)
            values = np.nan_to_num(values, nan=0.0)
            in_replicate_values.append(values)

        # Average IP and input replicates separately.
        ip_values = np.mean(ip_replicate_values, axis=0)
        in_values = np.mean(in_replicate_values, axis=0)

        # Calculate input-corrected signal.
        values = np.log1p(ip_values) - np.log1p(in_values)

        # Orient minus strand windows in 5' -> 3' direction.
        if row.strand == "-":
            values = values[::-1]

        extracted.append({
            "chr": row.chr,
            "start": start,
            "end": end,
            "strand": row.strand,
            "region_id": row.region_id,
            "block_number": row.block_number,
            "signal": values
        })

    for bw in plus_ip_bws + minus_ip_bws + plus_in_bws + minus_in_bws:
        bw.close()

    return(extracted)


# Set up argument parser.

parser = argparse.ArgumentParser()

# Add all arguments to parser.

parser.add_argument("--window_A", required=True)
parser.add_argument("--window_B", required=True)

parser.add_argument("--plus_A", nargs="+", required=True)
parser.add_argument("--minus_A", nargs="+", required=True)
parser.add_argument("--plus_IN_A", nargs="+", required=True)
parser.add_argument("--minus_IN_A", nargs="+", required=True)

parser.add_argument("--plus_B", nargs="+", required=True)
parser.add_argument("--minus_B", nargs="+", required=True)
parser.add_argument("--plus_IN_B", nargs="+", required=True)
parser.add_argument("--minus_IN_B", nargs="+", required=True)

parser.add_argument("--output", required=True)
parser.add_argument("--rbp_A", required=True)
parser.add_argument("--rbp_B", required=True)

# Load arguments.

args = parser.parse_args()

# Extract regions.

regions = load_regions([args.window_A, args.window_B], buffer=100, block_size=300)

# Extract input-corrected signal.

signal_A = extract_bigwig_signal(
    args.plus_A, args.minus_A, args.plus_IN_A, args.minus_IN_A, regions
)

signal_B = extract_bigwig_signal(
    args.plus_B, args.minus_B, args.plus_IN_B, args.minus_IN_B, regions
)

# Save output.

save_pair_dataset(
    signal_A, signal_B, args.output, args.rbp_A, args.rbp_B,
    infer_cell_type(args.rbp_A)
)