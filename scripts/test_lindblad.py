#!/usr/bin/env python
"""
This example scripts computes the hopping matrix of a 2D lattice with
power law (alpha) decay in the hopping amplitude
"""
import numpy as np
import csv
from mpi4py import MPI
import sys
sys.path.append("/home/daneel/gitrepos/lindblad_phasepoints/build/lib.linux-x86_64-2.7/") 
import lindblad_phasepoints as lb
 
def run_lb():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_size = 11
  l = lattice_size
  k = np.array([0.0,0.0])
  amp = 1.0
  frq = 0.0
  rad = 1.0
    
    
  #Initiate the parameters in object
  p = lb.ParamData(latsize=lattice_size, drv_amp=amp, drv_freq=frq, \
    cloud_rad = rad, kvec=k)

  #Initiate the DTWA system with the parameters and niter
  d = lb.BBGKY_System(p, comm)

  #Prepare the times
  t0 = 0.0
  ncyc = 1.0
  nsteps = 100
  times = np.linspace(t0, ncyc, nsteps)

  corrdata = d.evolve(times)
  
  if rank == 0:
    #Prepare the output files. One for each observable
    fname = "corr_time_" + "amp_" + str(amp) + "_frq" + str(frq) 
    fname += "_cldrad" + str(rad) + "kx_" + str(k[0]) + "ky_" 
    fname += str(k[1]) + "N_" + str(l) + ".txt"

    #Dump each observable to a separate file
    np.savetxt(fname, np.vstack((times, corrdata)).T, delimiter=' ')

if __name__ == '__main__':
  run_lb()