#!/usr/bin/env bash

STEREOGENE=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/Stereogene_benchmark/stereogene/src/StereoGene
CHROM=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/annotations/chrNameLength.txt
BEDGRAPH_DIR=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/Stereogene_benchmark/encode_bedgraphs
RESULTS_DIR=/tscc/lustre/ddn/scratch/kflanagan/RNA_protein_grammar/Stereogene_results/SLBP_HNRNPC_IN

cd "$BEDGRAPH_DIR/plus"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/plus" \
    profPath="$RESULTS_DIR/profiles" \
    confounder=SLBP_K562_ENCSR483NOP_IN_1.scaled.plus.bedGraph \
    SLBP_K562_ENCSR483NOP_IP_1.scaled.plus.bedGraph \
    HNRNPC_K562_ENCSR249ROI_IP_1.scaled.plus.bedGraph \
    resPath="$RESULTS_DIR/plus_1"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/plus" \
    profPath="$RESULTS_DIR/profiles" \
    confounder=SLBP_K562_ENCSR483NOP_IN_1.scaled.plus.bedGraph \
    SLBP_K562_ENCSR483NOP_IP_2.scaled.plus.bedGraph \
    HNRNPC_K562_ENCSR249ROI_IP_2.scaled.plus.bedGraph \
    resPath="$RESULTS_DIR/plus_2"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/minus" \
    profPath="$RESULTS_DIR/profiles" \
    confounder=SLBP_K562_ENCSR483NOP_IN_1.scaled.minus.bedGraph \
    SLBP_K562_ENCSR483NOP_IP_1.scaled.minus.bedGraph \
    HNRNPC_K562_ENCSR249ROI_IP_1.scaled.minus.bedGraph \
    resPath="$RESULTS_DIR/minus_1"

"$STEREOGENE" \
    chrom="$CHROM" \
    plotType=pdf \
    trackPath="$BEDGRAPH_DIR/minus" \
    profPath="$RESULTS_DIR/profiles" \
    confounder=SLBP_K562_ENCSR483NOP_IN_1.scaled.minus.bedGraph \
    SLBP_K562_ENCSR483NOP_IP_2.scaled.minus.bedGraph \
    HNRNPC_K562_ENCSR249ROI_IP_2.scaled.minus.bedGraph \
    resPath="$RESULTS_DIR/minus_2"