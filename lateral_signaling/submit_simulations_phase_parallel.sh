#!/bin/bash

#Submit this script with: sbatch  submit_simulations_phase_parallel.sh

#SBATCH --time=9:00:00   # walltime
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "LateralSignalingPhase"   # job name
#SBATCH --mail-user=pbhamidi@usc.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

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

CMD="python3 simulate_phase_run_many_parallel.py"
echo $CMD
$CMD

#======END================================= 


