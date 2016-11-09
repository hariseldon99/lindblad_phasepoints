#!/bin/bash
#########################################################################
## Name of my job
#PBS -N N6
#PBS -l walltime=00:15:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -o /mnt/lustre/users/aroy/mult_realizations/rho_0p03/stdout_N6.dat
#PBS -e /mnt/lustre/users/aroy/mult_realizations/rho_0p03/stdout_N6.dat
#PBS -j oe 
#########################################################################
##Number of nodes and procs/mpiprocs per node.
#PBS -l select=2:ncpus=24:mpiprocs=3:nodetype=haswell_reg
#PBS -q normal
#PBS -P PHYS0853
#########################################################################
##Send me email when my job aborts, begins, or ends
##PBS -m ea
##PBS -M daneel@utexas.edu
#########################################################################
SCRIPT="./largesize.py"
LATSIZE=14
#Set BLAS threads to 2 per MPI process
export OMP_NUM_THREADS=1
# Make sure I'm the only one that can read my output
umask 0077

cd $PBS_O_WORKDIR
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
##Now, run the code
BEGINTIME=$(date +"%s")
$HOME/anaconda2/bin/mpirun -x LD_LIBRARY_PATH -np $NO_OF_CORES -machinefile $PBS_NODEFILE  python -W ignore $SCRIPT -g $LATSIZE
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))
echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
