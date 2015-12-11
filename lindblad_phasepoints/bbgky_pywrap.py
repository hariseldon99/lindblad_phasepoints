#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import lindblad_bbgky as lbbgky

def lindblad_bbgky_pywrap(s, t, param):
   """
   Python wrapper to lindblad C bbgky module
   """
   #s[0:3N]  is the tensor s^l_\mu
   #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
   #Probably not wise to reshape b4 passing to a C routine.
   #By default, numpy arrays are contiguous, but reshaping...
   s = np.require(s, dtype=np.float64, \
     requirements=['A', 'O', 'W', 'C'])
   dsdt = np.zeros_like(s)
   dsdt = np.require(dsdt, dtype=np.float64, \
     requirements=['A', 'O', 'W', 'C'])
   lbbgky.bbgky(param.workspace, s, param.deltamat.flatten(), \
     param.gammamat.flatten(), (param.kr + param.drv_freq * t),\
       param.drv_amp, param.latsize,dsdt)
   return dsdt