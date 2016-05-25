#!/bin/bash
#########################################################################
## Name of my job
#PBS -N large_N
#PBS -l walltime=48:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -o /mnt/lustre/users/aroy/stdout.dat
#PBS -e /mnt/lustre/users/aroy/stderr.dat
#PBS -j oe 
#########################################################################
##Number of nodes and procs/mpiprocs per node.
#PBS -l select=3:ncpus=24:mpiprocs=18:nodetype=haswell_reg
#PBS -q normal
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@utexas.edu
#########################################################################
SCRIPT = "./large_N.py"
# Make sure I'm the only one that can read my output
umask 0077
#Set BLAS threads to 2 per MPI process
export OMP_NUM_THREADS=2
# Load the module system

cd $PBS_O_WORKDIR

#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################

#########################################################################
##Now, run the code
BEGINTIME=$(date +"%s")
mpirun -np $NO_OF_CORES -machinefile $PBS_NODEFILE  python -W ignore $SCRIPT 
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
