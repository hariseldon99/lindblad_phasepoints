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
  lattice_size = 12
  l = lattice_size
  amp = 1.0
  frq = 0.8
  rad = 5.0
    
    
  #Initiate the parameters in object
  p = lb.ParamData(latsize=lattice_size, drv_amp=amp, drv_freq=frq, \
    cloud_rad = rad, kvec_theta=0.0, kvec_phi=0.0)

  #Initiate the DTWA system with the parameters 
  d = lb.BBGKY_System(p, comm, verbose=True)

  #Prepare the times
  t0 = 0.0
  ncyc = 20.0
  nsteps = 1000
  times = np.linspace(t0, ncyc, nsteps)
  timestep = times[1]-times[0]
  corrdata, distribution = d.evolve(times)
  
  
  if rank == 0:
    freqs = np.fft.fftfreq(corrdata.size, d=timestep)
    spectrum = np.fft.fft(corrdata)
    #Prepare the output files. One for each observable
    fname = "corr_time_" + "amp_" + str(amp) + "_frq" + str(frq) 
    fname += "_cldrad" + str(rad) 
    fname += "N_" + str(l) + ".txt"

    #Dump each observable to a separate file
    np.savetxt(fname, np.vstack((np.abs(times), np.abs(corrdata))).T, delimiter=' ')
    fname = "spectrum_omega_" + "amp_" + str(amp) + "_frq" + str(frq) 
    fname += "_cldrad" + str(rad)  
    fname += "N_" + str(l) + ".txt"
    np.savetxt(fname, np.vstack((np.abs(freqs), np.abs(spectrum))).T, delimiter=' ')

if __name__ == '__main__':
  run_lb()
