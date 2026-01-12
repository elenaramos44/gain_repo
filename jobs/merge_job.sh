#!/bin/bash
#SBATCH --job-name=merge_wf_2307
#SBATCH --output=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/logs/merge_wf_2307_%A_%a.out
#SBATCH --error=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/logs/merge_wf_2307_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7   # ~1600 PMTs / 200 per job

module purge
module load Python/3.7.4-GCCcore-8.3.0

FOLDER="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307"
PMT_JSON="$FOLDER/pmts_list.json"

PMTS_PER_JOB=200

START=$(( SLURM_ARRAY_TASK_ID * PMTS_PER_JOB ))
END=$(( START + PMTS_PER_JOB - 1 ))

echo "[INFO] Merging PMTs ${START} → ${END}"

python3 /scratch/elena/WCTE_recovery/scripts/MERGE_separate_parts.py \
    --folder "$FOLDER" \
    --pmt-json "$PMT_JSON" \
    --start "$START" \
    --end "$END"
