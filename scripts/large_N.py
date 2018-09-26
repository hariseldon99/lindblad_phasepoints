#!/usr/bin/env python
from tabulate import tabulate
import numpy as np
from mpi4py import MPI
import lindblad_phasepoints as lb

def run_lb():
    #Main communicator and associated group
    comm = MPI.COMM_WORLD
    tc = 10 #Time chunks
    #Parameters
    amp = 8.0
    det = 0.0
    rho = 0.3
    momenta = np.array([[0.0, 0.0, 1.0]])
    num_atoms = 7
    #Prepare the times
    #Make sure that the time sample rate is orders of mag more than nyquist freq
    f = amp/(2.0 * np.pi)
    nyquist_freq = 2.5 * f
    sample_rate = nyquist_freq * 50.0
    interval = 1.0/sample_rate
    t0, tmax = 0.0, 15.0
    times = np.arange(t0, tmax, interval)
    #times_full = np.concatenate((-times[1:][::-1], times))
    timestep = times[1]-times[0]
    wt = MPI.Wtime()
    rank = comm.Get_rank()
    vol = num_atoms/rho
    r = pow((3./(4.*np.pi)) * vol,1./3.)
    #Initiate the parameters in object
    p = lb.ParamData(latsize=num_atoms, amplitude=amp, detuning=det,\
                  cloud_rad=r, kvecs=momenta)
    #Initiate the DTWA system with the parameters
    d = lb.BBGKY_System_Eqm(p, comm, verbose=False)
    (corrdata_f, distribution, atoms_info) = d.evolve(times, nchunks=tc)
    #Flip the system and rerun for backward time dynamics
    d.drv_amp, d.drv_freq = -d.drv_amp, -d.drv_freq
    d.deltamat = -d.deltamat
    (corrdata_b, distribution, atoms_info) = d.evolve(times, nchunks=tc)

    if rank == 0:
        for (count,data) in enumerate(corrdata_f):
            data = np.concatenate((corrdata_b[count][1:][::-1],\
                                                            corrdata_f[count]))
            fname = "corr_time_N_" + str(num_atoms) + "_omega_" +\
                                                str(amp) + "theta_0" + "_fd.txt"
            np.savetxt(fname, np.vstack((times, np.array(corrdata_f).real,\
                                np.array(corrdata_f).imag)).T, delimiter=' ')
            fname = "corr_time_N_" + str(num_atoms) + "_omega_" +\
                                                str(amp) + "theta_0" + "_bd.txt"
            np.savetxt(fname, np.vstack((times, np.array(corrdata_b).real,\
                                np.array(corrdata_b).imag)).T, delimiter=' ')

            freqs = np.fft.fftshift(np.fft.fftfreq(data.size, d=timestep))
            spectrum = np.fft.fftshift(np.fft.fft(data))
            #Dump each observable to a separate file
            fname = "spectrum_omega_N_" + str(num_atoms) + "amp_" + str(amp) +\
                                                            "_theta_0" + ".txt"
            np.savetxt(fname, np.vstack((2.0 * np.pi * freqs.real,\
                                               np.abs(spectrum))).T, delimiter=' ')
            print "size = ", num_atoms , "Walltime in secs = ", MPI.Wtime() - wt
            print "Atom positions:"
            #pprint(atoms_info, width=1)
            print tabulate(atoms_info, headers="keys")

if __name__ == '__main__':
    run_lb()