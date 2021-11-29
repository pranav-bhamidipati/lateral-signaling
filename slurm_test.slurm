#!/bin/bash

#SBATCH -J dask-worker
#SBATCH -t 00:20:00
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=pbhamidi@usc.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o /home/pbhamidi/scratch/lateral_signaling/slurm_out/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/pbhamidi/scratch/lateral_signaling/slurm_out/slurm.%N.%j.err # STDERR

echo "Started job"

sleep 60

echo "Trying Dask worker"

/central/home/pbhamidi/git/evomorph/env/bin/python -m distributed.cli.dask_worker tcp://131.215.148.23:41814 --nthreads 1 --memory-limit 1.86GiB --name dummy-name --nanny --death-timeout 60 --protocol tcp://

sleep 120
