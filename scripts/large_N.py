#!/usr/bin/env python
import sys
import numpy as np
from mpi4py import MPI
import lindblad_phasepoints as lb

def run_lb():
  #Main communicator and associated group  
  comm = MPI.COMM_WORLD
  group = comm.Get_group()  

  tc = 50 #Time chunks
  
  #Parameters
  amp = 8.0
  det = 0.0
  rho = 0.36
  momenta = np.array([[0.0, 0.0, 1.0]])
  #latsizes = np.array([7, 14, 21, 28, 35])
  latsizes = np.array([2, 4, 6])
  
  #Make sure that the main communicator is as big as the biggest lattice size
  if comm.Get_rank() == 0:
      if np.amax(latsizes) != comm.Get_size():
          print "Error, MPI Communicator too small." 
          print "Size is", comm.Get_size()
          print "Needed size", np.amax(latsizes)
          sys.exit()
          
  #Prepare the times. 
  #Make sure that the time sample rate is orders of mag more than nyquist freq
  f = amp/(2.0 * np.pi)
  nyquist_freq = 2.5 * f
  sample_rate = nyquist_freq * 500.0
  interval = 1.0/sample_rate
  t0, tmax = 0.0, 50.0
  times = np.arange(t0, tmax, interval)
  times_full = np.concatenate((-times[1:][::-1], times))
  timestep = times[1]-times[0]
  
  for l in latsizes:
      #Create a new communicator of size l from the main one
      newranks = np.arange(l)  
      newgroup = group.Incl(newranks)
      newcomm = comm.Create(newgroup)
      #Work only inside this communicator and not outside
      if comm.Get_rank() in newranks:
          rank = newcomm.Get_rank()          
          vol = l/rho
          r = pow((3./(4.*np.pi)) * vol,1./3.) 
   	    #Initiate the parameters in object
          p = lb.ParamData(latsize=l, amplitude=amp, detuning=det,\
          cloud_rad=r, kvecs=momenta)
          #Initiate the DTWA system with the parameters
          d = lb.BBGKY_System_Eqm(p, newcomm, verbose=True)
          (corrdata_f, distribution, atoms_info) = d.evolve(times, nchunks=tc)
          #Flip the system and rerun for backward time dynamics        
          d.drv_amp, d.drv_freq = -d.drv_amp, -d.drv_freq
          d.deltamat = -d.deltamat
          (corrdata_b, distribution, atoms_info) = d.evolve(times, nchunks=tc)
          
          if rank == 0:  
    		for (count,data) in enumerate(corrdata_f):
        		data = np.concatenate((corrdata_b[count][1:][::-1], corrdata_f[count]))
        		fname = "corr_time_N_" + str(l) + "_omega_" + str(amp) + "theta_0" + "_fd.txt"
        		np.savetxt(fname, np.vstack(\
                       		(times, np.array(corrdata_f).real, np.array(corrdata_f).imag)).T, delimiter=' ')
        		fname = "corr_time_N_" + str(l) + "_omega_" + str(amp) + "theta_0" + "_bd.txt"
        		np.savetxt(fname, np.vstack(\
                       		(times, np.array(corrdata_b).real, np.array(corrdata_b).imag)).T, delimiter=' ')

        		freqs = np.fft.fftshift(np.fft.fftfreq(data.size, d=timestep))
        		spectrum = np.fft.fftshift(np.fft.fft(data))
        		#Dump each observable to a separate file
        		np.savetxt(fname, np.vstack(\
                       		(times_full, data.real, data.imag)).T, delimiter=' ')
        		fname = "spectrum_omega_N_" + str(l) + "amp_" + str(amp) +\
                                			   		"_theta_0" + ".txt"
		        np.savetxt(fname, np.vstack((2.0 * np.pi * freqs.real,\
                		                           np.abs(spectrum))).T, delimiter=' ')
      
          newcomm.Free()
          newgroup.Free()

if __name__ == '__main__':
  run_lb()