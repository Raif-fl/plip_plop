#!/bin/bash
#SBATCH --job-name=encode_ml_inputs
#SBATCH --account=csd792
#SBATCH --partition=gold
#SBATCH --qos=hcg-csd792
#SBATCH --output=logs/encode_ml_inputs_%A_%a.out
#SBATCH --error=logs/encode_ml_inputs_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=1-30%10

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Starting task: ${SLURM_ARRAY_TASK_ID}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DIBS
module load samtools

SCRIPT="/tscc/nfs/home/kflanagan/projects/plip_plop/machine_learning_prototype/prepare_inputs.py"
COMPARISONS="/tscc/nfs/home/kflanagan/projects/plip_plop/machine_learning_prototype/encode_initial_30_comparisons.tsv"
OUTDIR="/tscc/lustre/ddn/scratch/kflanagan/plip_plop_results/processed_inputs"

mkdir -p logs
mkdir -p "$OUTDIR"

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMPARISONS")

IFS=$'\t' read -r \
    category \
    experiment_A \
    experiment_B \
    window_A \
    plus_1_A \
    plus_2_A \
    minus_1_A \
    minus_2_A \
    plus_IN_1_A \
    minus_IN_1_A \
    bam_1_A \
    bam_2_A \
    bam_IN_1_A \
    window_B \
    plus_1_B \
    plus_2_B \
    minus_1_B \
    minus_2_B \
    plus_IN_1_B \
    minus_IN_1_B \
    bam_1_B \
    bam_2_B \
    bam_IN_1_B <<< "$ROW"

cell_type=$(echo "$experiment_A" | cut -d'_' -f2)

echo "Processing ${experiment_A} vs ${experiment_B}"
echo "Cell type: ${cell_type}"

python "$SCRIPT" \
    --window_A "$window_A" \
    --window_B "$window_B" \
    --plus_A "$plus_1_A" "$plus_2_A" \
    --minus_A "$minus_1_A" "$minus_2_A" \
    --plus_IN_A "$plus_IN_1_A" \
    --minus_IN_A "$minus_IN_1_A" \
    --plus_B "$plus_1_B" "$plus_2_B" \
    --minus_B "$minus_1_B" "$minus_2_B" \
    --plus_IN_B "$plus_IN_1_B" \
    --minus_IN_B "$minus_IN_1_B" \
    --bam_A "$bam_1_A" "$bam_2_A" \
    --bam_IN_A "$bam_IN_1_A" \
    --bam_B "$bam_1_B" "$bam_2_B" \
    --bam_IN_B "$bam_IN_1_B" \
    --output "$OUTDIR/${experiment_A}_${experiment_B}.npz" \
    --rbp_A "$experiment_A" \
    --rbp_B "$experiment_B" \
    --cell_type "$cell_type" \
    --p 1 \
    --sigma 2

echo "Finished task: ${SLURM_ARRAY_TASK_ID}"