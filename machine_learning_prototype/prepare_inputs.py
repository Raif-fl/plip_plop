#!/usr/bin/env python3

import argparse
import os
import subprocess
import numpy as np
import pandas as pd
import pyBigWig
from scipy.ndimage import gaussian_filter1d


def get_mapped_reads(bam_file):
    result = subprocess.run(["samtools", "idxstats", bam_file], capture_output=True, text=True, check=True)
    return(sum(int(line.split("\t")[2]) for line in result.stdout.strip().split("\n") if line))


def calculate_rpm_pseudocount(library_sizes):
    library_sizes = np.atleast_1d(library_sizes).astype(float)
    return(np.mean(1e6 / library_sizes))


def load_regions(window_files, buffer=100, block_size=300):
    regions = pd.concat([pd.read_csv(f, sep="\t") for f in window_files], ignore_index=True)
    regions["start_buffered"] = (regions["start"] - buffer).clip(lower=0)
    regions["end_buffered"] = regions["end"] + buffer
    regions = regions.sort_values(["chr", "strand", "start_buffered", "end_buffered"])

    merged = []
    for (chrom, strand), group in regions.groupby(["chr", "strand"], sort=False):
        current_start = current_end = None
        for row in group.itertuples(index=False):
            if current_start is None:
                current_start, current_end = row.start_buffered, row.end_buffered
            elif row.start_buffered < current_end:
                current_end = max(current_end, row.end_buffered)
            else:
                merged.append({"chr": chrom, "strand": strand, "start_buffered": current_start, "end_buffered": current_end})
                current_start, current_end = row.start_buffered, row.end_buffered
        merged.append({"chr": chrom, "strand": strand, "start_buffered": current_start, "end_buffered": current_end})

    blocks = []
    for region_id, region in enumerate(merged):
        chrom, strand = region["chr"], region["strand"]
        start, end = region["start_buffered"], region["end_buffered"]

        if end - start < block_size:
            extra = block_size - (end - start)
            start = max(0, start - extra // 2)
            end = start + block_size

        block_start = start
        block_number = 0
        while block_start + block_size <= end:
            blocks.append({"chr": chrom, "strand": strand, "start_buffered": block_start,
                           "end_buffered": block_start + block_size, "region_id": region_id,
                           "block_number": block_number})
            block_start += block_size
            block_number += 1

        if block_start < end:
            blocks.append({"chr": chrom, "strand": strand, "start_buffered": end - block_size,
                           "end_buffered": end, "region_id": region_id, "block_number": block_number})

    return(pd.DataFrame(blocks))


def extract_bigwig_signal(plus_bigwigs, minus_bigwigs, plus_library_sizes, minus_library_sizes, regions):
    plus_bws = [pyBigWig.open(path) for path in plus_bigwigs]
    minus_bws = [pyBigWig.open(path) for path in minus_bigwigs]
    extracted = []

    try:
        for row in regions.itertuples(index=False):
            if row.strand == "+":
                bws, library_sizes = plus_bws, plus_library_sizes
            elif row.strand == "-":
                bws, library_sizes = minus_bws, minus_library_sizes
            else:
                continue

            if not all(row.chr in bw.chroms() for bw in bws):
                continue

            chrom_size = min(bw.chroms(row.chr) for bw in bws)
            start = max(0, row.start_buffered)
            end = min(row.end_buffered, chrom_size)
            if end - start != row.end_buffered - row.start_buffered:
                continue

            replicate_values = []
            for bw, library_size in zip(bws, library_sizes):
                values = bw.values(row.chr, start, end, numpy=True)
                values = np.nan_to_num(values, nan=0.0)
                replicate_values.append(values * (1e6 / library_size))

            signal = np.mean(replicate_values, axis=0)
            if row.strand == "-":
                signal = signal[::-1]

            extracted.append({"chr": row.chr, "start": start, "end": end, "strand": row.strand,
                              "region_id": row.region_id, "block_number": row.block_number, "signal": signal})
    finally:
        for bw in plus_bws + minus_bws:
            bw.close()

    return(extracted)


def abundance_weighted_enrichment(ip, in_signal, ip_pseudocount, in_pseudocount, p=1):
    mean_window_signal = (ip + in_signal).mean()
    k = p * mean_window_signal
    log_enrichment = np.log((ip + ip_pseudocount) / (in_signal + in_pseudocount))
    abundance = ip + in_signal
    abundance_weight = np.divide(abundance, abundance + k, out=np.zeros_like(abundance), where=(abundance + k) != 0)
    return(log_enrichment * abundance_weight)


def validate_args(args):
    groups = [
        (args.plus_A, args.minus_A, args.bam_A, "A IP"),
        (args.plus_IN_A, args.minus_IN_A, args.bam_IN_A, "A IN"),
        (args.plus_B, args.minus_B, args.bam_B, "B IP"),
        (args.plus_IN_B, args.minus_IN_B, args.bam_IN_B, "B IN")
    ]

    for plus, minus, bams, label in groups:
        if not (len(plus) == len(minus) == len(bams)):
            raise ValueError(f"{label} plus/minus bigWigs and BAM lists must have matching lengths.")


def process_comparison(args):
    validate_args(args)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    regions = load_regions([args.window_A, args.window_B], buffer=args.buffer, block_size=args.block_size)

    ip_A_library_sizes = [get_mapped_reads(path) for path in args.bam_A]
    in_A_library_sizes = [get_mapped_reads(path) for path in args.bam_IN_A]
    ip_B_library_sizes = [get_mapped_reads(path) for path in args.bam_B]
    in_B_library_sizes = [get_mapped_reads(path) for path in args.bam_IN_B]

    ip_A = extract_bigwig_signal(args.plus_A, args.minus_A, ip_A_library_sizes, ip_A_library_sizes, regions)
    in_A = extract_bigwig_signal(args.plus_IN_A, args.minus_IN_A, in_A_library_sizes, in_A_library_sizes, regions)
    ip_B = extract_bigwig_signal(args.plus_B, args.minus_B, ip_B_library_sizes, ip_B_library_sizes, regions)
    in_B = extract_bigwig_signal(args.plus_IN_B, args.minus_IN_B, in_B_library_sizes, in_B_library_sizes, regions)

    if not (len(ip_A) == len(in_A) == len(ip_B) == len(in_B)):
        raise ValueError("Extracted track groups contain different numbers of genomic blocks.")

    ip_A_pseudocount = calculate_rpm_pseudocount(ip_A_library_sizes)
    in_A_pseudocount = calculate_rpm_pseudocount(in_A_library_sizes)
    ip_B_pseudocount = calculate_rpm_pseudocount(ip_B_library_sizes)
    in_B_pseudocount = calculate_rpm_pseudocount(in_B_library_sizes)

    signals = []
    total_signal_A = []
    total_signal_B = []
    chrom, start, end, strand, region_id, block_number = [], [], [], [], [], []

    for a_ip, a_in, b_ip, b_in in zip(ip_A, in_A, ip_B, in_B):
        coordinates = [(x["chr"], x["start"], x["end"], x["strand"], x["region_id"], x["block_number"])
                       for x in [a_ip, a_in, b_ip, b_in]]
        if len(set(coordinates)) != 1:
            raise ValueError("Signal tracks do not contain matching genomic blocks.")

        total_signal_A.append(a_ip["signal"].sum())
        total_signal_B.append(b_ip["signal"].sum())

        signal_A = abundance_weighted_enrichment(a_ip["signal"], a_in["signal"], ip_A_pseudocount, in_A_pseudocount, p=args.p)
        signal_B = abundance_weighted_enrichment(b_ip["signal"], b_in["signal"], ip_B_pseudocount, in_B_pseudocount, p=args.p)
        signal_A = gaussian_filter1d(signal_A, sigma=args.sigma)
        signal_B = gaussian_filter1d(signal_B, sigma=args.sigma)

        signals.append(np.stack([signal_A, signal_B]))
        chrom.append(a_ip["chr"])
        start.append(a_ip["start"])
        end.append(a_ip["end"])
        strand.append(a_ip["strand"])
        region_id.append(a_ip["region_id"])
        block_number.append(a_ip["block_number"])

    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 3 or signals.shape[1:] != (2, args.block_size):
        raise ValueError(f"Unexpected signal shape: {signals.shape}")

    np.savez_compressed(
        args.output,
        signals=signals,
        chrom=np.asarray(chrom),
        start=np.asarray(start),
        end=np.asarray(end),
        strand=np.asarray(strand),
        region_id=np.asarray(region_id),
        block_number=np.asarray(block_number),
        rbp_A=np.asarray(args.rbp_A),
        rbp_B=np.asarray(args.rbp_B),
        cell_type=np.asarray(args.cell_type),
        p=np.asarray(args.p),
        sigma=np.asarray(args.sigma),
        ip_A_pseudocount=np.asarray(ip_A_pseudocount),
        in_A_pseudocount=np.asarray(in_A_pseudocount),
        ip_B_pseudocount=np.asarray(ip_B_pseudocount),
        in_B_pseudocount=np.asarray(in_B_pseudocount),
        total_signal_A=np.asarray(total_signal_A, dtype=np.float32),
        total_signal_B=np.asarray(total_signal_B, dtype=np.float32),
        total_signal=np.asarray(total_signal_A, dtype=np.float32) + np.asarray(total_signal_B, dtype=np.float32)
    )

    print(f"Saved {len(signals)} blocks to {args.output}")
    print(f"Signal shape: {signals.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IN-adjusted eCLIP signal windows for the RBP comparison model.")
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
    parser.add_argument("--bam_A", nargs="+", required=True)
    parser.add_argument("--bam_IN_A", nargs="+", required=True)
    parser.add_argument("--bam_B", nargs="+", required=True)
    parser.add_argument("--bam_IN_B", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rbp_A", required=True)
    parser.add_argument("--rbp_B", required=True)
    parser.add_argument("--cell_type", required=True)
    parser.add_argument("--buffer", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=300)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument("--sigma", type=float, default=2)
    return(parser.parse_args())


if __name__ == "__main__":
    process_comparison(parse_args())
