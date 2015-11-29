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
    N = self.latsize
 	
    if mpicomm.rank == root:
      if self.verbose:
          out = copy.copy(self)
          out.deltamn = 0.0
          pprint(vars(out), depth=2)

      #Build the gas cloud of atoms
      np.random.seed(seed)
      self.atoms = np.array(\
	[Atom(coords = r * np.random.random(3), index = i) \
	  for i in xrange(N)]) 
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
    if self.verbose and self.comm.rank == root:
      print("\nAtoms scattered to grid\n")
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

  def bare_correlations(self,init,array):
      return np.sum(fftconvolve(init,array))

  def field_correlations(self, t_output, sdata):
    """
    Compute the field correlations in
    times t_output wrt correlations at
    t_output[0]
    NOTE THAT THE SHAPE OF INPUT SDATA IS:
    (NATOMS, NALPHAS,NTIMES,3)
    """
    N = self.latsize
    norm = 8.0 * N
    phases = np.array([np.exp(-1j*self.kvec.dot(atom.coords))\
      for atom in self.atoms])
    phases_conj = np.conjugate(phases)
    corrs = np.zeros((nalphas,t_output.size), dtype=np.complex_)
    
    for alpha in xrange(nalphas):
        ek0_dagger = np.multiply(phases, sdata[:,alpha,0,0]) +  \
                (1j) * np.multiply(phases, sdata[:,alpha,0,1])
        for ti, t in np.ndenumerate(t_output):
            ekt = np.multiply(phases_conj, sdata[:,alpha,ti[0],0]) -\
                    (1j) * np.multiply(phases_conj, sdata[:,alpha,ti[0],1])
            corrs[alpha, ti[0]] = np.sum(fftconvolve(ek0_dagger, ekt))
    
    #Sum over alphas and normalize
    return np.sum(corrs,0)/norm
      
    
  def bbgky(self, time_info):
    """
    Evolves the BBGKY dynamics for selected phase points
    call with bbgky(t), where t is an array of times
    returns the field correlations wrt the initial field
    """
    N = self.latsize
    result = None
    if type(time_info).__module__ == np.__name__ :
      #An empty grid of size N X nalphas
      #Each element of this list is a dataset
      localdata = [[None for e in range(nalphas)] \
	for f in range(self.local_atoms.size)]
      for count, mth_atom in np.ndenumerate(self.local_atoms):
	(m, coord_m) = mth_atom.index, mth_atom.coords
        for alpha in xrange(nalphas):
	  a, c = self.initconds(alpha, m)
	  s_t = odeint(lindblad_bbgky_pywrap, \
		np.concatenate((a.flatten(),c.flatten())),\
		  time_info, args=(self,), Dfun=None)	    
	  am_t = s_t[:,0:3*N][:,m::N]
          localdata[count[0]][alpha] = am_t
  
      if self.verbose:
	  if self.comm.rank == root:
	    print("\nGathering all data to root now\n")
          fulldata , distribution = gather_to_root(self.comm, \
                  np.array(localdata), root=root)
          if self.comm.rank == root:
              print ("\nDistribution of atoms in grid\n")
              distro_table = tabulate(zip(np.arange(distribution.size),\
                      distribution), headers=["CPU rank","Local no. of atoms"],\
                        tablefmt="grid")
              print(distro_table)
              print ("\nStatistics of atoms in the volume\n")
              xlocs = np.array([atom.coords[0] for atom in self.atoms])
              ylocs = np.array([atom.coords[1] for atom in self.atoms])
              zlocs = np.array([atom.coords[2] for atom in self.atoms])
            
      else:
          fulldata, distribution = gather_to_root(self.comm, \
                  np.array(localdata), root=root)
          distribution = None
      if self.comm.rank == root:
	result = self.field_correlations(time_info, fulldata)
      
      return (result, distribution)

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
