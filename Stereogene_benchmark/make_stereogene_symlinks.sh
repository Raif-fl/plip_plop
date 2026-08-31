#!/usr/bin/env bash

INPUT_BASE="$1"
OUTPUT_BASE="$2"

mkdir -p "$OUTPUT_BASE/plus" "$OUTPUT_BASE/minus"

for strand in plus minus; do
    for f in "$INPUT_BASE/scaled/$strand"/*.bg; do
        base=$(basename "$f" .bg)
        ln -s "$f" "$OUTPUT_BASE/$strand/${base}.bedGraph"
    done
done