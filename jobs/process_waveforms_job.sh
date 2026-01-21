#!/bin/bash
#SBATCH --job-name=step1_wf_2306
#SBATCH --output=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2306/logs/step1_wf_2306_%A_%a.out
#SBATCH --error=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2306/logs/step1_wf_2306_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-7   # Adjust according to total PMTs / chunk size

# --- Load modules ---
module purge
module load Python/3.7.4-GCCcore-8.3.0

# --- Parameters ---
RUN=2306
CHUNK_SIZE=200  # PMTs per job
PULSE_WIDTH=790ps  # Change to 780ps or 790ps as needed

SIGNAL_DIR="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run${RUN}/${PULSE_WIDTH}"
OUT_DIR="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run${RUN}/${PULSE_WIDTH}/charges"

SCRIPT="/scratch/elena/WCTE_recovery/scripts/process_waveforms_to_charges.py"

mkdir -p $OUT_DIR

# --- Determine chunk ID ---
CHUNK_ID=${SLURM_ARRAY_TASK_ID}

echo "[INFO] STEP 1: Processing run=${RUN}, pulse=${PULSE_WIDTH}, chunk=${CHUNK_ID}"

python3 $SCRIPT \
    --signal-dir $SIGNAL_DIR \
    --out-dir $OUT_DIR \
    --chunk-id $CHUNK_ID \
    --chunk-size $CHUNK_SIZE

echo "[INFO] STEP 1 finished: chunk=${CHUNK_ID}"
