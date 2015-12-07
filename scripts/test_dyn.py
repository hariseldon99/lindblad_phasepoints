#!/usr/bin/env python
"""
This example scripts computes the hopping matrix of a 2D lattice with
power law (alpha) decay in the hopping amplitude
"""
import numpy as np
from pprint import pprint
import csv
from mpi4py import MPI
import sys
import lindblad_phasepoints as lb
 
def run_lb():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_size = 5
  l = lattice_size
  amp = 40.0
  det = 0.0
  r = 3.5
  steady_state_time = 8.0
  #Initiate the parameters in object
  p = lb.ParamData(latsize=lattice_size, amplitude=amp, detuning=det, cloud_rad=r, mtime=steady_state_time)

  #Initiate the DTWA system with the parameters 
  d = lb.BBGKY_System(p, comm, verbose=True)
  
  #Prepare the times
  t0 = 0.0
  ncyc = 10.0
  nsteps = 1000
  times = np.linspace(t0, ncyc, nsteps)
  timestep = times[1]-times[0]
  (corrdata, distribution, atoms_info) = d.evolve(times)
  
  
  if rank == 0:
    print " "
    print "Data of atoms in gas:"
    pprint(atoms_info)
    print "Distribution of atoms in grid"
    print distribution
    freqs = np.fft.fftfreq(corrdata.size, d=timestep)
    spectrum = np.fft.fft(corrdata)
    s = np.array_split(spectrum,2)[0]
    f = np.array_split(freqs,2)[0]
    #Prepare the output files. One for each observable
    fname = "corr_time_" + "amp_" + str(amp) + "_det" + str(det) 
    #fname += "_cldrad_" + str(rad) 
    fname += "_N_" + str(l) + ".txt"
    #Dump each observable to a separate file
    np.savetxt(fname, np.vstack((np.abs(times), corrdata.real, corrdata.imag)).T, delimiter=' ')

    fname = "spectrum_omega_" + "amp_" + str(amp) + "_det" + str(det) 
    #fname += "_cldrad_" + str(rad)  
    fname += "_N_" + str(l) + ".txt"
    np.savetxt(fname, np.vstack((np.abs(f), np.abs(s))).T, delimiter=' ')

if __name__ == '__main__':
  run_lb()
