#!/usr/bin/env python
import numpy as np
from pprint import pprint
import csv
from mpi4py import MPI
import sys
import lindblad_phasepoints as lb
from tabulate import tabulate

def run_lb():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_size = 5
  l = lattice_size
  amp = 40.0
  det = 3.0
  rad = 3.5
  thetas = np.array([0.0,np.pi/4.])
  kx = np.sin(thetas)
  ky = np.zeros(2)
  kz = np.cos(thetas)
  momenta = np.vstack((kx,ky,kz)).T

  #Initiate the parameters in object
  p = lb.ParamData(latsize=lattice_size, amplitude=amp, detuning=det, cloud_rad=rad, mtime=0.0, kvecs=momenta)

  #Initiate the DTWA system with the parameters 
  d = lb.BBGKY_System_Noneqm(p, comm, verbose=True)
  #Prepare the times
  t0 = 0.0
  ncyc = 1.0
  nsteps = 10
  times = np.linspace(t0, ncyc, nsteps)
  timestep = times[1]-times[0]
  (corrdata, distribution, atoms_info) = d.evolve(times, nchunks=1)
  if rank == 0:  
    print " "
    print "Data of atoms in gas:"
    print tabulate(atoms_info, headers="keys", tablefmt="fancy_grid")
    print "Distribution of atoms in grid"
    print distribution
    for (count,data) in enumerate(corrdata):
        freqs = np.fft.fftfreq(data.size, d=timestep)
    	spectrum = np.fft.fft(data)
    	s = np.array_split(spectrum,2)[0]
    	f = np.array_split(freqs,2)[0]
    	#Prepare the output files. One for each observable
    	fname = "corr_time_" + "amp_" + str(amp) + "_det_" + str(det) + "_theta_" + str(thetas[count])
    	fname += "_cldrad_" + str(rad) 
    	fname += "_N_" + str(l) + ".txt"
    	#Dump each observable to a separate file
    	np.savetxt(fname, np.vstack((np.abs(times), data.real, data.imag)).T, delimiter=' ')

    	fname = "spectrum_omega_" + "amp_" + str(amp) + "_det_" + str(det) + "_theta_" + str(thetas[count])
    	fname += "_cldrad_" + str(rad)  
    	fname += "_N_" + str(l) + ".txt"
    	np.savetxt(fname, np.vstack((np.abs(f), np.abs(s))).T, delimiter=' ')

if __name__ == '__main__':
  run_lb()
