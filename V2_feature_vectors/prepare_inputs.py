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

def load_regions(window_files, buffer=100, block_size=300, context=150):
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
            core_start = block_start
            core_end = block_start + block_size

            blocks.append({
                "chr": chrom, "strand": strand,
                "start_buffered": core_start, "end_buffered": core_end,
                "context_start": core_start - context,
                "context_end": core_end + context,
                "region_id": region_id, "block_number": block_number
            })

            block_start += block_size
            block_number += 1

        if block_start < end:
            core_start = end - block_size
            core_end = end

            blocks.append({
                "chr": chrom, "strand": strand,
                "start_buffered": core_start, "end_buffered": core_end,
                "context_start": core_start - context,
                "context_end": core_end + context,
                "region_id": region_id, "block_number": block_number
            })

    return pd.DataFrame(blocks)

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
            start = row.context_start
            end = row.context_end

            # Require the full context window.
            if start < 0 or end > chrom_size:
                continue

            replicate_signals = np.stack([
                np.nan_to_num(bw.values(row.chr, start, end, numpy=True), nan=0.0)
                for bw in bws
            ])

            # Orient minus-strand windows in 5' -> 3' direction.
            if row.strand == "-":
                replicate_signals = replicate_signals[:, ::-1]

            extracted.append({
                "chr": row.chr,
                "start": row.start_buffered,
                "end": row.end_buffered,
                "context_start": start,
                "context_end": end,
                "strand": row.strand,
                "region_id": row.region_id,
                "block_number": row.block_number,
                "replicate_signals": replicate_signals
            })

    finally:
        for bw in plus_bws + minus_bws:
            bw.close()

    return extracted

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

    regions = load_regions(
        [args.window_A, args.window_B],
        buffer=args.buffer,
        block_size=args.block_size,
        context=150
    )

    ip_A_library_sizes = [get_mapped_reads(path) for path in args.bam_A]
    in_A_library_sizes = [get_mapped_reads(path) for path in args.bam_IN_A]
    ip_B_library_sizes = [get_mapped_reads(path) for path in args.bam_B]
    in_B_library_sizes = [get_mapped_reads(path) for path in args.bam_IN_B]

    ip_A = extract_bigwig_signal(args.plus_A, args.minus_A, regions)
    in_A = extract_bigwig_signal(args.plus_IN_A, args.minus_IN_A, regions)
    ip_B = extract_bigwig_signal(args.plus_B, args.minus_B, regions)
    in_B = extract_bigwig_signal(args.plus_IN_B, args.minus_IN_B, regions)

    if not (len(ip_A) == len(in_A) == len(ip_B) == len(in_B)):
        raise ValueError("Extracted track groups contain different numbers of genomic blocks.")

    ip_signals_A = []
    in_signals_A = []
    ip_signals_B = []
    in_signals_B = []
    chrom, start, end, context_start, context_end, strand, region_id, block_number = [], [], [], [], [], [], [], []

    for a_ip, a_in, b_ip, b_in in zip(ip_A, in_A, ip_B, in_B):
        coordinates = [
            (x["chr"], x["start"], x["end"], x["context_start"], x["context_end"],
             x["strand"], x["region_id"], x["block_number"])
            for x in [a_ip, a_in, b_ip, b_in]
        ]

        if len(set(coordinates)) != 1:
            raise ValueError("Signal tracks do not contain matching genomic blocks.")

        ip_signals_A.append(a_ip["replicate_signals"])
        in_signals_A.append(a_in["replicate_signals"])
        ip_signals_B.append(b_ip["replicate_signals"])
        in_signals_B.append(b_in["replicate_signals"])

        chrom.append(a_ip["chr"])
        start.append(a_ip["start"])
        end.append(a_ip["end"])
        context_start.append(a_ip["context_start"])
        context_end.append(a_ip["context_end"])
        strand.append(a_ip["strand"])
        region_id.append(a_ip["region_id"])
        block_number.append(a_ip["block_number"])

    ip_signals_A = np.stack(ip_signals_A).astype(np.float32)
    in_signals_A = np.stack(in_signals_A).astype(np.float32)
    ip_signals_B = np.stack(ip_signals_B).astype(np.float32)
    in_signals_B = np.stack(in_signals_B).astype(np.float32)

    np.savez_compressed(
        args.output,
        chrom=np.asarray(chrom),
        start=np.asarray(start),
        end=np.asarray(end),
        context_start=np.asarray(context_start),
        context_end=np.asarray(context_end),
        strand=np.asarray(strand),
        region_id=np.asarray(region_id),
        block_number=np.asarray(block_number),

        rbp_A=np.asarray(args.rbp_A),
        rbp_B=np.asarray(args.rbp_B),
        cell_type=np.asarray(args.cell_type),

        ip_signals_A=ip_signals_A,
        in_signals_A=in_signals_A,
        ip_signals_B=ip_signals_B,
        in_signals_B=in_signals_B,

        ip_A_library_sizes=np.asarray(ip_A_library_sizes),
        in_A_library_sizes=np.asarray(in_A_library_sizes),
        ip_B_library_sizes=np.asarray(ip_B_library_sizes),
        in_B_library_sizes=np.asarray(in_B_library_sizes)
    )

    print(f"Saved {len(ip_signals_A)} blocks to {args.output}")
    print(f"A IP shape: {ip_signals_A.shape}")
    print(f"A IN shape: {in_signals_A.shape}")
    print(f"B IP shape: {ip_signals_B.shape}")
    print(f"B IN shape: {in_signals_B.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare replicate-level eCLIP signal windows for relational feature extraction.")
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
    return(parser.parse_args())


if __name__ == "__main__":
    process_comparison(parse_args())
