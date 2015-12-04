#!/usr/bin/env python
from __future__ import division, print_function
import sys
from mpi4py import MPI
from redirect_stdout import stdout_redirected
import copy
import numpy as np
from scipy.integrate import odeint
from pprint import pprint
from numpy.linalg import norm

from consts import *
from classes import *

#Try to import mkl if it is available
try:
  import mkl
  mkl_avail = True
except ImportError:
  mkl_avail = False

#Try to import lorenzo's optimized bbgky module, if available
import lindblad_bbgky as lbbgky

def kdel(i,j):
  if (i==j):
    return 1.
  else:
    return 0.
  
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

  def __init__(self, params, mpicomm, atoms=None,verbose=False):
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
       atoms		= numpy array of atom objects. If 'None', then builds then
			  atoms randomly
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
    N = self.latsize
 	
    if mpicomm.rank == root:
      if self.verbose:
          out = copy.copy(self)
          out.deltamn = 0.0
          pprint(vars(out), depth=2)

      #Build the gas cloud of atoms
      np.random.seed(seed)
      if atoms == None:
	self.atoms = np.array(\
	  [Atom(coords = r * (2.0 * np.random.random(3)-1.0), index = i) \
	    for i in xrange(N)])
      #TODO: Impose interparticle spacing restrictions
      elif type(atoms).__module__ == np.__name__:
	if atoms.size >= N:
	  self.atoms = atoms[0:N]
	else:
	  print("Error. Gas of atoms bigger than specified size")
	  sys.exit(0)
      else:
	self.atoms = atoms
    else:
      self.atoms = None
      
    #Create a workspace for mean field evaluaions
    self.workspace = np.zeros(2*(3*N+9*N*N))
    self.workspace = np.require(self.workspace, \
            dtype=np.float64, requirements=['A', 'O', 'W', 'C'])
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
    self.deltamat = np.zeros((N,N))
    self.gammamat = np.eye(N)
    for i in xrange(N):
      r_i = self.atoms[i].coords
      j=i+1
      while(j<N):
	r_j = self.atoms[j].coords
	arg = norm(r_i-r_j)
	self.deltamat[i,j] = -0.5 * np.cos(arg)/arg
	self.gammamat[i,j] = np.sin(arg)/arg
	j+=1
    self.deltamat = self.deltamat + self.deltamat.T
    self.gammamat = self.gammamat + self.gammamat.T
    self.kr = np.array([self.kvec.dot(atom_mu.coords) \
      for atom_mu in self.atoms])
    
  def initconds(self, alpha, lattice_index):
    N = self.latsize
    m = lattice_index
    a = np.zeros((3,self.latsize))
    a[2] = np.ones(N)
    a[:,m] = rvecs[alpha]
    c = np.zeros((3,3,self.latsize, self.latsize))
    return a, c
  
  def field_correlations(self, t_output, sdata, atom):
    """
    Compute the field correlations in
    times t_output wrt correlations at
    t_output[0]
    """
    N = self.latsize 
    (m, coord_m) = atom.index, atom.coords
    phase_m = np.exp(-1j*self.kvec.dot(coord_m))
    init_m = sdata[0,0:N][m] + (1j) * sdata[0,N:2*N][m] 
    phases_conj = np.array([np.exp(1j*self.kvec.dot(atom.coords))\
      for atom in self.atoms])
    return init_m * phase_m * \
      ((sdata[:,0:N]- (1j)*sdata[:,N:2*N]).dot(phases_conj))
      
    
  def bbgky(self, time_info):
    """
    Evolves the BBGKY dynamics for selected phase points
    call with bbgky(t), where t is an array of times
    returns the field correlations wrt the initial field
    """
    N = self.latsize

    if type(time_info).__module__ == np.__name__ :
      #An empty grid of size N X nalphas
      #Each element of this list is a dataset
      localdata = [None for f in range(self.local_atoms.size)]
      for tpl, mth_atom in np.ndenumerate(self.local_atoms):
	(count,) = tpl
	(m, coord_m) = mth_atom.index, mth_atom.coords
	
	corrs_summedover_alpha = \
	  np.zeros(time_info.size, dtype=np.complex_)
        for alpha in xrange(nalphas):
	  a, c = self.initconds(alpha, m)
	  s_t = odeint(lindblad_bbgky_pywrap, \
		np.concatenate((a.flatten(),c.flatten())),\
		  time_info, args=(self,), Dfun=None)	    
	  corrs_summedover_alpha += \
	    self.field_correlations(time_info, s_t[:,0:3*N], mth_atom)
	
	localdata[count] = corrs_summedover_alpha
      localsum_data = np.sum(np.array(localdata), axis=0)

      alldata = self.comm.reduce(localsum_data, root=root)
      
      if self.comm.rank == root:
	
	alldata = np.array(alldata)/self.corr_norm
	allsizes = np.zeros(self.comm.size)
	distrib_atoms = np.zeros_like(allsizes)
      else:
	allsizes = None
	distrib_atoms = None

      distrib_atoms = \
	self.comm.gather(self.local_atoms.size, distrib_atoms, root=0)
      return (alldata, distrib_atoms, \
	[atom.__dict__ for atom in self.atoms])

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
      An tuple object that contains
	t. A numpy array of field correlations at time wrt field at time[0]
    """
    return self.bbgky(time_info)
