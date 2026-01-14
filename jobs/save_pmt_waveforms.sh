#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=wf_2307_1
#SBATCH --output=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/logs/wf_2307_1_%A_%a.out
#SBATCH --error=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/logs/wf_2307_1_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-999%100               #total tasks=number of lines in "tasks.txt" - 1 is 2714; up to 50 array tasks at once


# Load modules 
module purge
module load foss/2019b
module load Python/3.7.4-GCCcore-8.3.0
source /scratch/elena/elena_wcsim/build/env_wcsim.sh
export PYTHONPATH=/scratch/$USER/python-libs:$PYTHONPATH


# Parameters
RUN=2307
CHUNK_SIZE=250
OUTDIR=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run${RUN}
SCRIPT=/scratch/elena/WCTE_recovery/scripts/save_pmt_waveforms.py
TASK_FILE=/scratch/elena/WCTE_recovery/scripts/tasks.txt


mkdir -p $OUTDIR  #make sure output dir exists 


# Determine part and chunk
TASK_ID=${SLURM_ARRAY_TASK_ID}
read PART CHUNK < <(sed -n "$((TASK_ID+1))p" $TASK_FILE)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running run=${RUN} part=${PART} chunk=${CHUNK}"
echo "TASK_ID=${TASK_ID}, LINE=$(sed -n "$((TASK_ID+1))p" $TASK_FILE)"


# run Python script 
python3 $SCRIPT \
    --run $RUN \
    --part $PART \
    --chunk-id $CHUNK \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR \
    --base-path /scratch/elena/WCTE_recovery/PMTs_calib_root_files \
    --verbose


echo "Task finished: run=${RUN} part=${PART} chunk=${CHUNK}"
