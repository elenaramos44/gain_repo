#!/bin/bash
#SBATCH --job-name=step2_fit_2306_770
#SBATCH --output=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2306/logs/step2_fit_2306_770_%A_%a.out
#SBATCH --error=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2306/logs/step2_fit_2306_770_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=0-7   # Same as STEP 1: total PMTs / chunk size

# --- Load modules ---
module purge
module load Python/3.7.4-GCCcore-8.3.0

# --- Parameters ---
RUN=2306
CHUNK_SIZE=200
PULSE_WIDTH=770ps  # Change accordingly
IN_DIR="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run${RUN}/${PULSE_WIDTH}/charges"
OUT_DIR="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run${RUN}/${PULSE_WIDTH}/results"

SCRIPT="/scratch/elena/WCTE_recovery/scripts/fit_charges_pmts.py"

mkdir -p $OUT_DIR

# --- Determine chunk ID ---
CHUNK_ID=${SLURM_ARRAY_TASK_ID}

echo "[INFO] STEP 2: Fitting charges for run=${RUN}, pulse=${PULSE_WIDTH}, chunk=${CHUNK_ID}"
#echo "[INFO] STEP 2: Fitting charges for run=${RUN}, chunk=${CHUNK_ID}"

python3 $SCRIPT \
    --in-dir $IN_DIR \
    --out-dir $OUT_DIR \
    --chunk-id $CHUNK_ID \
    --chunk-size $CHUNK_SIZE

echo "[INFO] STEP 2 finished: chunk=${CHUNK_ID}"
