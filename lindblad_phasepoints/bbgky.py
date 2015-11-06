#!/usr/bin/env python
from __future__ import division, print_function

from mpi4py import MPI
from redirect_stdout import stdout_redirected
import copy
import numpy as np
from scipy.integrate import odeint
from pprint import pprint
from tabulate import tabulate
from scipy.signal import fftconvolve


from consts import *
from classes import *

#Try to import mkl if it is available
try:
  import mkl
  mkl_avail = True
except ImportError:
  mkl_avail = False

#Try to import lorenzo's optimized bbgky module, if available
#import lindblad_bbgky as lb

      
def correlations(t_output, s, params):
  N = params.latsize
  """
  Compute \sum_{ij}<sx_i sx_j> -<sx>^2.
  """
  
  return None

def lindblad_bbgky_pywrap(s, t, param):
    """
    Python wrapper to lindblad C bbgky module
    """
    #s[0:3N]  is the tensor s^l_\mu
    #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
    #Probably not wise to reshape b4 passing to a C routine.
    #By default, numpy arrays are contiguous, but reshaping...
    dsdt = np.zeros_like(s)
    lb.bbgky(param.workspace, s, param.jmat.flatten(), \
      param.jvec, param.hvec, param.latsize,param.norm,dsdt)
    return dsdt
 
class BBGKY_System:
  """
    Class that creates the BBGKY system.
    
       Introduction:  
	This class instantiates an object encapsulating the optimized 
	BBGKY problem. It has methods that sample the trajectories
	from phase points and execute the BBGKY dynamics where the rhs 
	of the dynamics uses optimized C code. These methods call integrators 
	from scipy and time-evolve all the sampled initial conditions.
  """

  def __init__(self, params, mpicomm, verbose=False):
    """
    Initiates an instance of the Dtwa_System class. Copies parameters
    over from an instance of ParamData and stores precalculated objects .
    
       Usage:
       d = Dtwa_System(Paramdata, MPI_COMMUNICATOR, verbose=True)
       
       Parameters:
       Paramdata 	= An instance of the class "ParamData". 
			  See the relevant docs
       MPI_COMMUNICATOR = The MPI communicator that distributes the samples
			  to parallel processes. Set to MPI_COMM_SELF if 
			  running serially
       n_t		= Number of initial conditions to sample randomly 
			  from the discreet spin phase space. Defaults to
			  2000.
       verbose		= Boolean for choosing verbose outputs. Setting 
			  to 'True' dumps verbose output to stdout, which
			  consists of full output from the integrator. 
			  Defaults to 'False'.			  
			  
      Return value: 
      An object that stores all the parameters above.
    """

    self.__dict__.update(params.__dict__)
    self.comm = mpicomm
    #Booleans for verbosity and for calculating site data
    self.verbose = verbose
    r = self.cloud_rad
    N = params.latsize
    if mpicomm.rank == root:
      #Create a workspace for mean field evaluaions
      self.workspace = np.zeros(3*N+9*N*N)
      self.workspace = np.require(self.workspace, \
	dtype=np.float64, requirements=['A', 'O', 'W', 'C'])
      #Build the gas cloud of atoms
      self.atoms = np.array(\
	[Atom(coords = r * np.random.random(2), index = i) \
	  for i in xrange(N)]) 
    else:
      self.workspace = None
      self.atoms = None
      
    self.workspace = mpicomm.bcast(self.workspace, root=root)  
    self.atoms = mpicomm.bcast(self.atoms, root=root)  
    #Scatter local copies of the atoms
    if mpicomm.rank == root:
      sendbuf = np.array_split(atoms,mpicomm.size)
      local_size = np.array([spl.size for spl in atoms])
    else:
      sendbuf = None
      local_size = None
    local_size = mpicomm.scatter(local_size, root = root)
    local_atoms = np.empty(local_size, dtype="float64")
    local_atoms = mpicomm.scatter(sendbuf, root = root)
    
  def bbgky(self, time_info):
    return None
     
  def evolve(self, time_info):
    """
    This function calls the lsode 'odeint' integrator from scipy package
    to evolve all the sampled initial conditions in time. 
    The lsode integrator controls integrator method and 
    actual time steps adaptively. Verbosiy levels are decided during the
    instantiation of this class. After the integration is complete, each 
    process returns the mth site data to root. The root then computes spectral
    properties as output
    
    
       Usage:
       data = d.evolve(times)
       
       Required parameters:
       times 		= Time information. There are 2 options: 
			  1. A 3-tuple (t0, t1, steps), where t0(1) is the 
			      initial (final) time, and steps are the number
			      of time steps that are in the output. 
			  2. A list or numpy array with the times entered
			      manually.
			      
			      Note that the integrator method and the actual step sizes
			      are controlled internally by the integrator. 
			      See the relevant docs for scipy.integrate.odeint.

      Return value: 
      An tuple object that contains:
	1. The times, bound to the method t_output
	2. A numpy array of vectors (only) at the times
    """
    return self.bbgky(time_info, sampling)
 
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #Initiate the parameters in object
    p = ParamData(latsize=101)
    #Initiate the DTWA system with the parameters and niter
    d = BBGKY_System(p, comm)
    data = d.evolve((0.0, 1.0, 1000))