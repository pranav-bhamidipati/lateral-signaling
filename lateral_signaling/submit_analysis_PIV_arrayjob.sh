#!/bin/bash --login

#Submit this script with: sbatch submit_analysis_PIV_arrayjob.sh

#SBATCH --time=04:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per job in array (i.e. tasks)
#SBATCH --array=0-791
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "SynNotch_PIV_ArrayJob"   # job name
#SBATCH --mail-user=pbhamidi@usc.edu   # email address
#SBATCH --mail-type=ALL

#SBATCH -o /home/pbhamidi/slurm_out/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/pbhamidi/slurm_out/slurm.%N.%j.err # STDERR

 
#======START===============================

module purge

eval "$(conda shell.bash hook)"
mamba init
source ~/.bashrc
source ~/mambaforge/etc/profile.d/mamba.sh

echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "A total of $SLURM_NTASKS tasks is used"
echo "Environment Variables"
env
echo ""

echo "Activating mamba environment"

mamba activate ~/group/envs/lateral_signaling
echo ""

CMD="python3 analyze_PIV_arrayjob.py"
echo $CMD
$CMD


#======END================================= 


