#!/usr/bin/env bash

STEREOGENE=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/Stereogene_benchmark/stereogene/src/StereoGene
CHROM=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/annotations/chrNameLength.txt
BEDGRAPH_DIR=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/Stereogene_benchmark/ADAR_bedgraphs
RESULTS_DIR=/tscc/lustre/ddn/scratch/kflanagan/RNA_protein_grammar/Stereogene_results/ADARs

cd "$BEDGRAPH_DIR/plus"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/plus" \
    profPath="$RESULTS_DIR/profiles" \
    ADARp110_IP_1.scaled.plus.bedGraph \
    ADARp150_IP_1.scaled.plus.bedGraph \
    resPath="$RESULTS_DIR/plus_1"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/plus" \
    profPath="$RESULTS_DIR/profiles" \
    ADARp110_IP_2.scaled.plus.bedGraph \
    ADARp150_IP_2.scaled.plus.bedGraph \
    resPath="$RESULTS_DIR/plus_2"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/minus" \
    profPath="$RESULTS_DIR/profiles" \
    ADARp110_IP_1.scaled.minus.bedGraph \
    ADARp150_IP_1.scaled.minus.bedGraph \
    resPath="$RESULTS_DIR/minus_1"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/minus" \
    profPath="$RESULTS_DIR/profiles" \
    ADARp110_IP_2.scaled.minus.bedGraph \
    ADARp150_IP_2.scaled.minus.bedGraph \
    resPath="$RESULTS_DIR/minus_2"