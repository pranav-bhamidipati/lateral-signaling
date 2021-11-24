#!/bin/bash
#SBATCH --job-name="LateralSignalingPhase"
#SBATCH --ntasks=64                # How many threads to request
#SBATCH --time=15:00:00               
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pbhamidi@usc.edu  

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
conda deactivate
conda activate ~/git/evomorph/env
echo ""

CMD="python3 lsig_phase_run_many_parallel.py"
echo $CMD
$CMD

#======END================================= 


