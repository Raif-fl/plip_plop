#!/usr/bin/env bash

STEREOGENE=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/Stereogene_benchmark/stereogene/src/StereoGene
CHROM=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/annotations/chrNameLength.txt
BEDGRAPH_DIR=/tscc/projects/ps-yeolab3/kflanagan/RNA_protein_grammar/Stereogene_benchmark/TAU_m6A_bedgraphs
RESULTS_DIR=/tscc/lustre/ddn/scratch/kflanagan/RNA_protein_grammar/Stereogene_results/TAU_m6a_IN

cd "$BEDGRAPH_DIR"

"$STEREOGENE" \
    chrom="$CHROM" \
    outRes=BOTH \
    plotType=pdf \
    L_LC=0 \
    R_LC=0 \
    trackPath="$BEDGRAPH_DIR" \
    profPath="$RESULTS_DIR/profiles" \
    confounder=1328_MAPTKD_WTTau_3XFLAG_IN_1.scaled.bedGraph \
    1328_MAPTKD_WTTau_3XFLAG_IP_1.scaled.bedGraph \
    m6A_IP_1.bedGraph \
    resPath="$RESULTS_DIR/rep_1" \
    -lc 

"$STEREOGENE" \
    chrom="$CHROM" \
    outRes=BOTH \
    plotType=pdf \
    L_LC=0 \
    R_LC=0 \
    trackPath="$BEDGRAPH_DIR" \
    profPath="$RESULTS_DIR/profiles" \
    confounder=1328_MAPTKD_WTTau_3XFLAG_IN_2.scaled.bedGraph \
    1328_MAPTKD_WTTau_3XFLAG_IP_2.scaled.bedGraph \
    m6A_IP_2.bedGraph \
    resPath="$RESULTS_DIR/rep_2" \
    -lc 