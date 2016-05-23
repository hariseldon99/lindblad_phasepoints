#!/usr/bin/env python
import numpy as np
from pprint import pprint
from mpi4py import MPI
import lindblad_phasepoints as lb

def run_lb():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Parameters
  lattice_size = 7
  l = lattice_size
  amps = np.array([3.0, 5.0, 8.0, 10.0, 20.0, 30.0, 40.0])
  #Prepare the times
  f = np.amax(amps)/(2.0 * np.pi)
  nyquist_freq = 2.5 * f
  sample_rate = nyquist_freq * 500.0
  interval = 1.0/sample_rate
  t0, tmax = 0.0, 50.0
  times = np.arange(t0, tmax, interval)
  times_full = np.concatenate((-times[1:][::-1], times))
  timestep = times[1]-times[0]
  det = 0.0
  rho = 1.0
  vol = l/rho
  r = pow((3./(4.*np.pi)) * vol,1./3.) 

  momenta = np.array([[0.0,0.0, 1.0]])

  pos = np.loadtxt("positions_N_7_scale_0.3_dt_0.005_T_50.0_Omega0_40_fd.dat",delimiter='\t')
  atoms_atscale = \
               np.array([lb.Atom(coords = pos[i], index = i) for i in xrange(l)])
  for amp in amps:
	#Initiate the parameters in object
  	p = lb.ParamData(latsize=lattice_size, amplitude=amp, detuning=det, cloud_rad=r, kvecs=momenta)
  	#Initiate the DTWA system with the parameters 
  	d = lb.BBGKY_System_Eqm(p, comm, atoms=atoms_atscale, verbose=True)
  	(corrdata_f, distribution, atoms_info) = d.evolve(times, nchunks=50)
  	#Flip the system and rerun for backward time dynamics        
  	d.drv_amp, d.drv_freq = -d.drv_amp, -d.drv_freq
  	d.deltamat = -d.deltamat
  	(corrdata_b, distribution, atoms_info) = d.evolve(times, nchunks=50)
    
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
        		#Prepare the output files. One for each observable
        		fname = "corr_time_" + "_theta_0" + ".txt"
        		#Dump each observable to a separate file
        		np.savetxt(fname, np.vstack(\
                       		(times_full, data.real, data.imag)).T, delimiter=' ')
        		fname = "spectrum_omega_" + "amp_" + str(amp) +\
                                			   "_theta_0" + ".txt"
		        np.savetxt(fname, np.vstack((2.0 * np.pi * freqs.real,\
                		                           np.abs(spectrum))).T, delimiter=' ')

if __name__ == '__main__':
  run_lb()
