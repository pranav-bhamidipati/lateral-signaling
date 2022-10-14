#!/bin/bash

#Submit this script with: sbatch submit_simulations_phase_arrayjob.sh

#SBATCH --time=09:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per job in array (i.e. tasks)
#SBATCH --array=0-575
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "LateralSignalingPhase"   # job name
#SBATCH --mail-user=pbhamidi@usc.edu   # email address
#SBATCH --mail-type=ALL

#SBATCH -o /home/pbhamidi/scratch/lateral_signaling/slurm_out/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/pbhamidi/scratch/lateral_signaling/slurm_out/slurm.%N.%j.err # STDERR

 
#======START===============================

source ~/.bashrc
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "A total of $SLURM_NTASKS tasks is used"
echo "Environment Variables"
env
echo ""

echo "Activating conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ~/git/evomorph/env
echo ""

CMD="python3 simulate_phase_run_many_array.py"
echo $CMD
$CMD

#======END================================= 


