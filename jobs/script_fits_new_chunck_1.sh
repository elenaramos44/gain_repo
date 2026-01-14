#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=final_fit_2307
#SBATCH --output=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/logs/final_fit_2307_%A_%a.out
#SBATCH --error=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/logs/final_fit_2307_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-15   # Ajusta según total PMTs / chunk-size

#----------------- LOAD ENVIRONMENT -----------------
source /scratch/elena/elena_wcsim/build/env_wcsim.sh


#----------------- RUN PYTHON SCRIPT -----------------
python3 /scratch/elena/WCTE_recovery/scripts/NEW_script_fits_chunck_1.py \
    --pattern "card*_slot*_ch*_pos*.npz" \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 100
