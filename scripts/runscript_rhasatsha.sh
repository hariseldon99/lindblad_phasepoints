#!/bin/bash
#########################################################################
## Name of my job
#PBS -N lindblad_largeN
#PBS -l walltime=48:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -o stdout.txt
#PBS -e stderr.txt
#PBS -j oe
#########################################################################
##Number of nodes and procs per node.
##The ib at the end means infiniband. Use that or else MPI gets confused 
##with ethernet
#PBS -l select=2:ncpus=36:mpiprocs=18,place=scatter
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################
SCRIPT="./large_N.py"

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=2
#########################################################################
##Make a list of allocated nodes(cores)
##Note that if multiple jobs run in same directory, use different names
##for example, add on jobid nmber.
#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################

#########################################################################
##Now run my prog
module load dot
BEGINTIME=$(date +"%s")
##Now, run the code
mpirun -np $NO_OF_CORES --hostfile ${PBS_NODEFILE} \
      python -W ignore $SCRIPT 
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
