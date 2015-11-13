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
from numpy.linalg import norm

from consts import *
from classes import *
from default_gather import *

#Try to import mkl if it is available
try:
  import mkl
  mkl_avail = True
except ImportError:
  mkl_avail = False

#Try to import lorenzo's optimized bbgky module, if available
import lindblad_bbgky as lb

def lindblad_bbgky_test_native(s, t, param):
  N = param.latsize
  stensor = s[0:3*N].reshape(3,N)
  gtensor = s[3*N:].reshape(3,3,N,N)
  dsdt = -stensor
  dgdt = -gtensor
  return np.concatenate((dsdt.flatten(), dgdt.flatten()))

def lindblad_bbgky_test_pywrap(s, t, param):
    """
    Python wrapper to lindblad C bbgky module
    """
    #s[0:3N]  is the tensor s^l_\mu
    #G = s[3*N:].reshape(3,3,N,N) is the tensor g^{ab}_{\mu\nu}.
    #Probably not wise to reshape b4 passing to a C routine.
    #By default, numpy arrays are contiguous, but reshaping...
    dsdt = np.zeros_like(s)
    dtkr = np.array([(param.drv_freq * t) + \
      param.kvec.dot(atom_mu.coords) for atom_mu in param.local_atoms])
    
    lb.bbgky(param.workspace, s, param.deltamat.flatten(), \
      param.gammamat.flatten(), dtkr, param.drv_amp, param.latsize,dsdt)
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
      sendbuf = np.array_split(self.atoms,mpicomm.size)
      local_size = np.array([spl.size for spl in sendbuf])
    else:
      sendbuf = None
      local_size = None
    local_size = mpicomm.scatter(local_size, root = root)
    self.local_atoms = np.empty(local_size, dtype="float64")
    self.local_atoms = mpicomm.scatter(sendbuf, root = root)
    kmag = norm(params.kvec)
    self.deltamat = np.zeros((N,N))
    self.gammamat = np.eye(N)
    for i in xrange(N):
      r_i = self.atoms[i].coords
      j=i+1
      while(j<N):
	r_j = self.atoms[j].coords
	arg = kmag * norm(r_i-r_j)
	if np.abs(kmag)< threshold:
	  self.deltamat[i,j] = -0.5 
	  self.gammamat[i,j] = 0.0
	else:
	  self.deltamat[i,j] = -0.5 * np.cos(arg)/arg
	  self.gammamat[i,j] = np.sin(arg)/arg
	j+=1
    self.deltamat = self.deltamat + self.deltamat.T
    self.gammamat = self.gammamat + self.gammamat.T
    
    
  def initconds(self, alpha, lattice_index):
    N = self.latsize
    m = lattice_index
    a = np.zeros((3,self.latsize))
    a[2] = np.ones(N)
    a[:,m] = rvecs[alpha]
    c = np.zeros((3,3,self.latsize, self.latsize))
    return a, c

  def correlations(self, t_output, sdata):
    N = self.latsize
    """
    Compute \sum_{ij}<sx_i sx_j> -<sx>^2.
    """
    posvecs = np.array([atom.coords for atom in self.atoms])
    phases = np.exp(np.array([1j*self.kvec.dot(r) for r in posvecs]))
    print(phases.shape, posvecs.shape)
    c0 = np.multiply(phases, sdata[0,0,:]) +  \
      (1j) * np.multiply(phases, sdata[0,1,:])
    corrs = []
    for t in t_output:
      i = np.where(t_output == t)
      ct = np.multiply(phases, sdata[i,0,:]) +  \
      (1j) * np.multiply(phases, sdata[i,1,:])
      c = fftconvolve(ct, c0)
      corrs.append(c/(8.0*pow(2,N-1)))
    
    return np.array(corrs)
    
  def bbgky(self, time_info):
    """
    Evolves the BBGKY dynamics for selected phase points
    call with bbgky(t), where t is an array of times
    """
    N = self.latsize
    result = None
    if type(time_info).__module__ == np.__name__ :
      for mth_atom in self.local_atoms:
	(m, coord_m) = mth_atom.extract()
	data = []
	for alpha in xrange(nalphas):
	  a, c = self.initconds(alpha, m)
	  
	  s_t = odeint(lindblad_bbgky_test_pywrap, \
		np.concatenate((a.flatten(),c.flatten())),\
		  time_info, args=(self,), Dfun=None)
	  am_t = s_t[:,0:3*self.latsize][:,m:-1:N]
	  data.append(am_t)
	afm_t = np.sum(data,axis=1).flatten()  
      
      fulldata = gather_to_root(self.comm, MPI.DOUBLE, np.array(data), root=root)
      if self.comm.rank == root:
	result = self.correlations(time_info, fulldata)
    return result
     
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
    return self.bbgky(time_info)
 
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #Initiate the parameters in object
    p = ParamData(latsize=101)
    #Initiate the DTWA system with the parameters and niter
    d = BBGKY_System(p, comm)
    data = d.evolve((0.0, 1.0, 1000))