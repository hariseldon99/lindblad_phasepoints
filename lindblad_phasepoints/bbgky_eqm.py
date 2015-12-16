#!/usr/bin/env python
from __future__ import division, print_function
import sys
from mpi4py import MPI
from reductions import Intracomm
import copy
import numpy as np
from scipy.integrate import odeint
from pprint import pprint
from numpy.linalg import norm

from consts import *
from classes import *
from bbgky_pywrap import *
from generate_coord import *

#Try to import mkl if available
try:
  import mkl
  mkl_avail = True
except ImportError:
  mkl_avail = False
  
#Try to import progressbars if available
try:
    import progressbar
    pbar_avail = True
except ImportError:
    pbar_avail = False
    
class BBGKY_System_Eqm:
  """
    Class that creates the BBGKY system.
    
       Introduction:  
	This class instantiates an object encapsulating the optimized 
	BBGKY problem. It has methods that sample the trajectories
	from phase points and execute the BBGKY dynamics where the rhs 
	of the dynamics uses optimized C code. These methods call integrators 
	from scipy and time-evolve all the sampled initial conditions.
	This class is for the equilibrium spectra.
  """

  def __init__(self, params, mpicomm, atoms=None,verbose=False):
    """
    Initiates an instance of the BBGKY_System_Noneqm class. Copies 
    parameters over from an instance of ParamData and stores 
    precalculated objects .
    
       Usage:
       d = BBGKY_System_Eqm(Paramdata, MPI_COMMUNICATOR, verbose=True)
       
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
    self.mkl_avail = mkl_avail
    self.pbar_avail = pbar_avail
    
    if self.comm.rank == root:
      if self.verbose:
          out = copy.copy(self)
          out.deltamn = 0.0
	  pprint(vars(out), depth=2)
      #Build the gas cloud of atoms
      if atoms == None:
	c, self.mindist  = generate_coordinates(self.latsize,\
	  min = self.intpt_spacing, max = self.cloud_rad,\
	    verbose=self.verbose)
	if self.verbose:
	  print("\nDone. Minimum distance between atoms = ", self.mindist)
	self.atoms = np.array(\
	  [Atom(coords = c[i], index = i) for i in xrange(N)])
	self.kr_incident = np.array([\
	  self.kvec_incident.dot(atom_mu.coords) \
	  for atom_mu in self.atoms])
      elif type(atoms).__module__ == np.__name__:
	if atoms.size >= N:
	  self.atoms = atoms[0:N]
	else:
	  print("Error. Gas of atoms smaller than specified size")
	  sys.exit(0)
      else:
	self.atoms = atoms
	self.kr_incident = np.array([\
	  self.kvec_incident.dot(atom_mu.coords) \
	  for atom_mu in self.atoms])
    else:
      self.atoms = None
      self.kr_incident = None
      
    #Create a workspace for mean field evaluaions
    self.workspace = np.zeros(2*(3*N+9*N*N))
    self.workspace = np.require(self.workspace, \
            dtype=np.float64, requirements=['A', 'O', 'W', 'C'])
    self.atoms = self.comm.bcast(self.atoms, root=root)  
    self.kr_incident = self.comm.bcast(self.kr_incident, root=root)  
    #Scatter local copies of the atoms
    if self.comm.rank == root:
      sendbuf = np.array_split(self.atoms,self.comm.size)
      local_size = np.array([spl.size for spl in sendbuf])
    else:
      sendbuf = None
      local_size = None
    local_size = self.comm.scatter(local_size, root = root)
    self.local_atoms = np.empty(local_size, dtype="float64")
    self.local_atoms = self.comm.scatter(sendbuf, root = root)
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
    
  #CHECK THIS!!!!!  
  def initconds(self, steady_state ,alpha, lattice_index, a=None):
    """
    Set the 4 initial conditions in eqns 63 and 64 of writeup
    'a' is the vector superscript of rho, can run from:
    None, 0 (x), 1(y), 2(z)
    """
    N = self.latsize
    m = lattice_index
    ic = steady_state
    if a == None:
      ic[0:3*N][:,m] = rvecs[alpha]
    else:
      #Get \tilde{\rho}_{ss}
      spins_mat = ic[0:3*N].reshape(3,N)
      stensor = ic[3*N:].reshape(3,3,N,N)
      #First, shift from connected correlations to bare correlations
      spinprod = np.einsum("mi,nj->mnij",spins_mat, spins_mat)
      stensor += spinprod
      #stensor are the bare correlations now
    if a in [0,1,2]:
      #Now change to tilde
      denr = (1.0 + spins_mat[a,m])
      stensor = (stensor + spins_mat[a,m] * spinprod)/denr
      spins_mat = (spins_mat + stensor[a,:,m,:])/denr
      #Now shift back to connected correlations
      stensor -= np.einsum("mi,nj->mnij",spins_mat, spins_mat)
      ic = np.concatenate((spins_mat.flatten(),stensor.flatten()))
    else:
      ic = None
      
    return ic 
  
  def initconds_bbgkyonly(self):
    """
    Set all spins to [0,0,1]
    i.e. z-polarized spins
    """
    N = self.latsize
    a = np.zeros((3,self.latsize))
    a[2] = np.ones(N)
    c = np.zeros((3,3,self.latsize, self.latsize))
    return a, c
  
  def field_correlations(self, t_output, atom, *allsdata):
    """
    Compute the field correlations in
    times t_output wrt correlations near
    self.mtime
    """
    N = self.latsize
    #Adjust the value mtime to the nearest value in the input array
    init_arg = np.abs(t_output - self.mtime).argmin()
    (m, coord_m) = atom.index, atom.coords
    #DO THIS! IMPLEMENT eqn (62)
    for sdata in allsdata:
      pass
    
    return None
      
  def get_norm(self, rho):
    """
    Normalizes the density matrix according to
    eqn (65) in the writeup
    """
    #DO THIS!!!
    return 1.0

    
  def bbgky_eqm(self, time_info):
    """
    Evolves the BBGKY dynamics for selected phase points
    call with bbgky(t), where t is an array of times
    returns the "nonequilibrium" field correlations 
    i.e. correlations w.r.t. the initial field
    """
    N = self.latsize

    if type(time_info).__module__ == np.__name__ :
      #An empty grid of size N X nalphas
      #Each element of this list is a dataset
      localdata = [[None for f in range(self.local_atoms.size)] \
	for kvec in self.kvecs]
      if pbar_avail:
	if self.comm.rank == root and self.verbose: 
	  pbar_max = \
	    self.kvecs.shape[0] * self.local_atoms.size * nalphas - 1
	  bar = progressbar.ProgressBar(widgets=widgets_bbgky,\
	    max_value=pbar_max, redirect_stdout=False)
	   
      bar_pos = 0
      if self.verbose and pbar_avail and self.comm.rank == root:
	  bar.update(bar_pos)

      #CHECK THIS!!!
      #Set the initconds to evaluate the steady state using bare bbgky
      #Also, get the norm from eq (65) in the writeup
      #Let the root processor do this, then broadcast it
      if self.comm.rank == root:
	a, c = self.initconds_bbgkyonly()
	s_t = odeint(lindblad_bbgky_pywrap, \
	      np.concatenate((a.flatten(),c.flatten())),\
		time_info, args=(self,), Dfun=None)
	rho_ss = s_t[-1]
	self.corr_norm = self.get_norm(rho_ss)
      else:
	rho_ss = None
	self.corr_norm = None
      rho_ss = self.comm.bcast(rho_ss, root=root)
      self.corr_norm = self.comm.bcast(self.corr_norm, root=root)
      
      for tpl, mth_atom in np.ndenumerate(self.local_atoms):
	(atom_count,) = tpl
	(m, coord_m) = mth_atom.index, mth_atom.coords
	corrs_summedover_alpha = \
	  np.zeros((self.kvecs.shape[0], time_info.size), \
	    dtype=np.complex_)
	for alpha in xrange(nalphas):
	  
	  #CHECK THIS!!!
	  #Set \rho_{ss,\not\mu}A_{\alpha\mu} at t=0
	  s0 = self.initconds(rho_ss, alpha, m, a=None)
	  rho_ss_A = odeint(lindblad_bbgky_pywrap, s0, time_info, \
	    args=(self,), Dfun=None)
	  #Set \tilde{\rho}^x_{ss,\not\mu}A_{\alpha\mu} at t=0
	  s0 = self.initconds(rho_ss, alpha, m, a=0)
	  rho_ss_x_A = odeint(lindblad_bbgky_pywrap, s0, time_info, \
	    args=(self,), Dfun=None)
	  #Set \tilde{\rho}^y_{ss,\not\mu}A_{\alpha\mu} at t=0
	  s0 = self.initconds(rho_ss, alpha, m, a=1)
	  rho_ss_y_A = odeint(lindblad_bbgky_pywrap, s0, time_info, \
	    args=(self,), Dfun=None)
	  #Set \tilde{\rho}^z_{ss,\not\mu}A_{\alpha\mu} at t=0
	  s0 = self.initconds(rho_ss, alpha, m, a=2)
	  rho_ss_z_A = odeint(lindblad_bbgky_pywrap, s0, time_info, \
	    args=(self,), Dfun=None)
	  
	  for kcount in xrange(self.kvecs.shape[0]):
	    self.kvec = self.kvecs[kcount]
	    corrs_summedover_alpha[kcount] += \
	      self.field_correlations(time_info,mth_atom, \
		rho_ss[0:3*N], rho_ss_A[0:3*N], \
		  rho_ss_x_A[0:3*N], rho_ss_y_A[0:3*N], \
		    rho_ss_z_A[0:3*N])
	    if self.verbose and pbar_avail and self.comm.rank == root:
	      bar_pos = kcount*(self.local_atoms.size * nalphas) + \
		atom_count * nalphas + alpha
	      bar.update(bar_pos)
	    localdata[kcount][atom_count] = corrs_summedover_alpha[kcount]
            bar_pos += 1
	    
      duplicate_comm = Intracomm(self.comm)
      alldata = [None for i in self.kvecs]
      for kcount in xrange(self.kvecs.shape[0]):
	localsum_data = np.sum(np.array(localdata[kcount]), axis=0)
	alldata[kcount] = duplicate_comm.reduce(localsum_data, root=root)
	
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
       times 		=  Time information. Must be a list or numpy array 
			   with the times entered. Need not be uniform
			      
			   Note that the integrator method and the actual step sizes
			   are controlled internally by the integrator. 
			   See the relevant docs for scipy.integrate.odeint.

      Return value: 
      An tuple object (data, distrib, atomdata) that contains
	data		=  A numpy array of field correlations at time wrt field at 
			   mtime (provided in params). The shape is (times, # of kvecs 
			   provided in params) so data[j] are the correlations for 
			   kvecs[j] at all the times provided
	distrib		=  A numpy array where distrib[i] is the number of atoms 
			   processed by MPI rank i (for debugging purposes)
	atomdata	=  A dictionary containing the indices and positions
			   of all the atoms
    """
    return self.bbgky_eqm(time_info)
