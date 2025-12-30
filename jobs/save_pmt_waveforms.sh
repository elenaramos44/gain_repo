#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=wf_2307_5
#SBATCH --output=wf_2307_5_%A_%a.out
#SBATCH --error=wf_2307_5_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1760-2199  # Total tasks = número de líneas en tasks.txt

module purge
module load foss/2019b
module load Python/3.7.4-GCCcore-8.3.0
source /scratch/elena/elena_wcsim/build/env_wcsim.sh
export PYTHONPATH=/scratch/$USER/python-libs:$PYTHONPATH


RUN=2307
CHUNK_SIZE=250
OUTDIR=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/waveforms_including_position
SCRIPT=/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/save_pmt_waveforms.py
TASK_FILE=tasks.txt


TASK_ID=${SLURM_ARRAY_TASK_ID}

# Leer part y chunk desde tasks.txt
read PART CHUNK < <(sed -n "$((TASK_ID+1))p" $TASK_FILE)

echo "Processing run=$RUN part=$PART chunk=$CHUNK"

python3 $SCRIPT \
    --run $RUN \
    --part $PART \
    --chunk-id $CHUNK \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR
