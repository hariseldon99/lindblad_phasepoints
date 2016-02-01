#!/usr/bin/env python
from __future__ import division, print_function
import sys
from mpi4py import MPI
from reductions import Intracomm
import copy
import numpy as np
from scipy.integrate import ode, odeint
from pprint import pprint
from numpy.linalg import norm
from itertools import product

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
    self.corr_norm = 16.0 * self.latsize
    
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
    self.atoms = mpicomm.bcast(self.atoms, root=root)  
    self.kr_incident = mpicomm.bcast(self.kr_incident, root=root)  
    #Scatter local copies of the atoms
    if self.comm.rank == root:
      sendbuf = np.array_split(self.atoms,mpicomm.size)
      local_size = np.array([spl.size for spl in sendbuf])
    else:
      sendbuf = None
      local_size = None
    local_size = mpicomm.scatter(local_size, root = root)
    self.local_atoms = np.empty(local_size, dtype="float64")
    self.local_atoms = mpicomm.scatter(sendbuf, root = root)
    self.deltamat = np.zeros((N,N))
    self.gammamat = np.zeros_like(self.deltamat)
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
  
  def field_correlations(self, t_output, alpha, sdata, atom):
    """
    Compute the field correlations in
    times t_output wrt correlations near
    self.mtime
    """
    N = self.latsize
    (m, coord_m) = atom.index, atom.coords
    #DO EQUATION 62 HERE

      
    
  def bbgky_noneqm(self, times):
    """
    Evolves the BBGKY dynamics for selected phase points
    call with bbgky(t), where t is an array of times
    returns the "equilibrium" field correlations 
    i.e. correlations w.r.t. the final steady state
    """
    N = self.latsize

    if type(times).__module__ == np.__name__ :
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
      for tpl, mth_atom in np.ndenumerate(self.local_atoms):
	(atom_count,) = tpl
	(m, coord_m) = mth_atom.index, mth_atom.coords
	corrs_summedover_alpha = \
	  np.zeros((self.kvecs.shape[0], times.size), \
	    dtype=np.complex_)
	for alpha in xrange(nalphas):
	  #CHANGE THIS TO 4 EVOLUTIONS AND UPDATES!!
	  s_t = odeint(lindblad_bbgky_pywrap, \
	    mth_atom.state[alpha], times, args=(self,), Dfun=None)
	  #Update the final state
	  self.local_atoms[atom_count].state[alpha] = s_t[-1] 
	  for kcount in xrange(self.kvecs.shape[0]):
	    self.kvec = self.kvecs[kcount]
	    corrs_summedover_alpha[kcount] += \
	      self.field_correlations(times, alpha, s_t[:,0:3*N], mth_atom)
	    if self.verbose and pbar_avail and self.comm.rank == root:
	      bar.update(bar_pos)
	    localdata[kcount][atom_count] = corrs_summedover_alpha[kcount]
            bar_pos += 1
	    
      duplicate_comm = Intracomm(self.comm)
      alldata = np.array([None for i in self.kvecs])
      for kcount in xrange(self.kvecs.shape[0]):
	localsum_data = np.sum(np.array(localdata[kcount]), axis=0)
	if self.comm.size == 1:
	  alldata[kcount] = localsum_data
	else:
	  alldata[kcount] = duplicate_comm.reduce(localsum_data, root=root)
	  
      if self.comm.rank == root:
	alldata /= self.corr_norm
      return alldata

  def evolve(self, time_info, nchunks=1):
    """
    This function calls the lsode 'odeint' integrator from scipy package
    to evolve all the sampled initial conditions in time. 
    The lsode integrator controls integrator method and 
    actual time steps adaptively. Verbosiy levels are decided during the
    instantiation of this class. After the integration is complete, each 
    process returns the mth site data to root. The root then computes spectral
    properties as output
    
    
       Usage:
       data = d.evolve(times, nchunks=100)
       
       Required parameters:
       times 		=  Time information. Must be a list or numpy array 
			   with the times entered. Need not be uniform
			      
			   Note that the integrator method and the actual step sizes
			   are controlled internally by the integrator. 
			   See the relevant docs for scipy.integrate.odeint.
      nchunks		=  Number of chunks. This divides "times" into nchunks 
			      parts and runs them independently to conserve memory.
			      Defaults to 1.
      Return value: 
      An tuple object (data, distrib, atomdata) that contains
	data		=  A numpy array of field correlations at time wrt field at 
			   initial time. The shape is (times, # of kvecs 
			   provided in params) so data[j] are the correlations for 
			   kvecs[j] at all the times provided
	distrib		=  A numpy array where distrib[i] is the number of atoms 
			   processed by MPI rank i (for debugging purposes)
	atomdata	=  A dictionary containing the indices and positions
			   of all the atoms
    """
    N = self.latsize
    #Initial time
    self.mtime = time_info[0]
    #Empty list     
    outdata = []
    times_split = np.array_split(time_info, nchunks)
    t_sizes = np.array([t.size for t in times_split])
	#The refstate is the final steady state after a long time,
	#internally set in "consts.py"
	#Only have root do this, then broadcast
	if self.comm.rank == root:  
		a = np.zeros((3,self.latsize))
		a[2] = np.ones(self.latsize)
		c = np.zeros((3,3,self.latsize, self.latsize))
		initstate = np.concatenate((a.flatten(), c.flatten()))
		self.steady_state = np.zeros_like(initstate)
		r = ode(lindblad_bbgky_pywrap_ode).set_integrator(int_method)
		r.set_initial_value(initstate, steady_state_init_time).set_f_params(self)
		while r.successful() and r.t < steady_state_final_time:
			self.steady_state = r.integrate(r.t+steady_state_dt)
                #Disconnect the correlations in the steady_state 
                #i.e. store s^{ab}_{ij} instead of g^{ab}_{ij}
                self.steady_state[3*N:].reshape(3,3,N,N) + np.einsum("ai,bj->abij",self.steady_state[0:3*N].reshape(3,N), self.steady_state[0:3*N].reshape(3,N)) 
	else:
		self.steady_state = None
	self.steady_state = self.comm.bcast(self.steady_state, root=0)  
	 
    #Set the initial conditions and the reference states
    for (alpha, mth_atom) in product(np.arange(nalphas), self.local_atoms):
      m = mth_atom.index
      
      #This is gonna have 4 states each as per method 3 in Lorenzo's writeup
      mth_atom.state[alpha] = [None, None, None, None]
      
      #Eq 65 in Lorenzo's writeup
      mth_atom.state[alpha][0] = self.steady_state
      #Tracing out single particle terms
      mth_atom.state[alpha][0][0:3*N].reshape(3,N)[:,m] = rvecs[alpha]
      #Tracing out 2 particle terms
      mth_atom.state[alpha][0][3*N:].reshape(3,3,N,N)[:,:,np.full(N,m),range(N)] = 0.0
      mth_atom.state[alpha][0][3*N:].reshape(3,3,N,N)[:,:,range(N), np.full(N,m)] = 0.0

      
      #Eq 64 in lorenzo's writeup for a = 'x'
      #Applying the 'tilde' transformation in Eq 60 of Lorenzo's writeup
      mth_atom.state[alpha][1] = np.zeros_like(self.steady_state)
      mth_atom.state[alpha][1][0:3*N] = self.steady_state[0:3*N] + self.steady_state[3*N].reshape(3,3,N,N)[0,:,m,:]
      #APPLY TILDE TRANSFORMATION IN EQ 61. FINISH USING EQ 17.
      mth_atom.state[alpha][1]/= 1.0 + self.steady_state[0:3*N].reshape(3,N)[0,m]
      #Tracing out single particle terms
      mth_atom.state[alpha][1][0:3*N].reshape(3,N)[:,m] = rvecs[alpha]
      #Tracing out 2 particle terms
      mth_atom.state[alpha][1][3*N:].reshape(3,3,N,N)[:,:,np.full(N,m),range(N)] = 0.0
      mth_atom.state[alpha][1][3*N:].reshape(3,3,N,N)[:,:,range(N), np.full(N,m)] = 0.0
	  
	  #SIMILARLY DO EQ 63 FOR a= y and z
          #PUT THE TRACE-OUTS IN A SEPARATE FUNCTION, SINCE THEY'RE ALL THE FRIGGIN' SAME
	  #ALSO CALCULATE AND STORE THE NORMS
	 
      mth_atom.refstate[alpha] = rvecs[alpha]
      
    for i, times in enumerate(times_split):
      if i < len(times_split)-1:
	times[-1] = times_split[i+1][0]
      outdata.append(self.bbgky_noneqm(times))
     
    if self.comm.rank == root:
	allsizes = np.zeros(self.comm.size)
	distrib_atoms = np.zeros_like(allsizes)
    else:
	allsizes = None
	distrib_atoms = None
    distrib_atoms = \
      self.comm.gather(self.local_atoms.size, distrib_atoms, root=0)
      
    if self.comm.rank == root:
      #This is an ugly hack, but it works
      temp = np.asarray(outdata).T
      outdata = []
      for data in temp:
	outdata.append(np.concatenate(tuple(data)))
      return (outdata, distrib_atoms, \
	[atom.__dict__ for atom in self.atoms])
    else:
      return (None, None, None)
