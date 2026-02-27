#!/bin/bash
# Merge per-task CSVs from part (b) array job into a single file.
# Usage: bash student/slurm/collect_results.sh

RESULTS_DIR="${1:-results/bench_b}"
OUT="results/bench_part_b.csv"

echo "model,mode,avg_ms,std_ms,min_ms,max_ms" > "$OUT"
cat "${RESULTS_DIR}"/*.csv >> "$OUT"

echo "Merged $(ls "${RESULTS_DIR}"/*.csv | wc -l) results -> $OUT"
