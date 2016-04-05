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
from itertools import product, combinations

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
    
  def disconnect(self,state):
	  """
	  Disconnect the correlations in a state 
	  i.e. store s^{ab}_{ij} instead of g^{ab}_{ij}
	  """
	  N = self.latsize
	  state[3*N:] = (state[3*N:].reshape(3,3,N,N) + \
		np.einsum("ai,bj->abij", state[0:3*N].reshape(3,N), state[0:3*N].reshape(3,N))).flatten()
			
  def reconnect(self,state):
	  """
	  Reconnect the correlations in a disconnected state 
	  i.e. store g^{ab}_{ij} instead of s^{ab}_{ij}
	  """
	  N = self.latsize
	  state[3*N:] = (state[3*N:].reshape(3,3,N,N) - \
		np.einsum("ai,bj->abij", state[0:3*N].reshape(3,N), state[0:3*N].reshape(3,N))).flatten()
			 
  def traceout_1p (self, state, m, alpha):
	  """
	  Trace out all the mth, and substitute with 
	  the alpha^th phase point operator
	  """
	  N = self.latsize
	  state[0:3*N].reshape(3,N)[:,m] = rvecs[alpha]
	  
  def traceout_2p (self, state, m, alpha):
   """
	  Trace out all 2-particle operators in state
	  except for the mth, and substitute with 
	  the alpha^th phase point operator
        THIS NEEDS CHECKING
   """
   N = self.latsize
   state[3*N:].reshape(3,3,N,N)[:,:,np.arange(m),m] = 0.0
   state[3*N:].reshape(3,3,N,N)[:,:,m,np.arange(m+1,N)] = 0.0

  def tilde_trans (self, state, a, m):
   """
	  Tilde transforms the input state as defined in Eq 60-61 of 
	  Lorenzo's writeup 
	  This is always done on a disconnected state
	  a = x,y,z i.e. 0,1,2
   """
   N = self.latsize
   state_1p = state[0:3*N].reshape(3,N)
   state_2p = state[3*N:].reshape(3,3,N,N)
   #reconnect the disconnected correlators
   state2p_conn = state_2p - np.einsum("am,bn->abmn",state_1p,state_1p)
   newstate_1p = state_1p
   newstate_2p = state_2p
   denr = 1.0 + state_1p[a,m]
   #This is eq 60
   newstate_1p += state_2p[a,:,m,:]
   newstate_1p/= denr
   #From Eq 17 truncating LHS to 0
   state_3p = np.einsum("am,bcng->abcmng",state_1p, state2p_conn)
   state_3p += np.einsum("bn,acmg->abcmng",state_1p, state2p_conn)
   state_3p += np.einsum("cg,abmn->abcmng",state_1p, state2p_conn)
   state_3p += np.einsum("am,bn,cg->abcmng",state_1p,state_1p,state_1p)
   #This is eq 61
   newstate_2p += state_3p[a,:,:,m,:,:]
   newstate_2p /= denr
   state = np.concatenate((newstate_1p.flatten(), newstate_2p.flatten()))
	  
  def field_correlations(self, alpha, r_t, atom):
    """
    Compute the equilibrium field correlations in
    times t_output 
    """
    N = self.latsize
    (m, coord_m) = atom.index, atom.coords
    #EQUATION 63 BELOW	      
    lx_m = rvecs[alpha][0]
    ly_m = rvecs[alpha][1]
    lz_m = rvecs[alpha][2]
    sx_m = self.steady_state[0:N][m]
    sy_m = self.steady_state[N:2*N][m]
    sz_m = self.steady_state[2*N:3*N][m]
    lxx_nm = r_t[0][:,0:N]
    lyx_nm = r_t[0][:,N:2*N]
    lyy_nm = r_t[1][:,N:2*N]
    lxy_nm = r_t[1][:,0:N]
    lxz_nm = r_t[2][:,0:N]
    lyz_nm = r_t[2][:,0:2*N]
    lx_nm = r_t[3][:,0:N]
    ly_nm = r_t[3][:,N:2*N]
    corrs_summedover_n = (1.+sx_m)*(1.-lz_m)*np.sum(lxx_nm-(1j)*lyx_nm, axis=1)
    corrs_summedover_n += (1.+sy_m)*(1.-lz_m)*np.sum(lyy_nm+(1j)*lxy_nm, axis=1)
    lxz_nm_sum = np.sum(lxz_nm, axis=1) 
    lyz_nm_sum = np.sum(lyz_nm, axis=1)
    corrs_summedover_n += (1.+sz_m)*(lx_m * lxz_nm_sum + ly_m * lyz_nm_sum)
    corrs_summedover_n += (1j)*(1.+sz_m)*(ly_m * lxz_nm_sum - lx_m * lyz_nm_sum)
    corrs_summedover_n += (lz_m-1)*((1.+1j) * np.sum(lx_nm,axis=1) +\
        (1-1j) * np.sum(ly_nm, axis=1))        
    return corrs_summedover_n

  def bbgky_noneqm(self, times):
    """
    Evolves the BBGKY dynamics for selected phase points
    call with bbgky(t), where t is an array of times
    returns the "equilibrium" field correlations 
    i.e. correlations w.r.t. the final steady state
    """
    N = self.latsize
    r_t = [None, None, None, None]
    #Evaluate the norm according to eq 76
    if self.comm.rank == root:
        norm_1 = N+np.sum(self.steady_state[2*N:3*N])
        sxxpsyy = self.steady_state[3*N:].reshape(3,3,N,N)[0,0,:,:] +\
            self.steady_state[3*N:].reshape(3,3,N,N)[1,1,:,:]
        sxymsyx = self.steady_state[3*N:].reshape(3,3,N,N)[0,1,:,:] -\
            self.steady_state[3*N:].reshape(3,3,N,N)[1,0,:,:]
        self.norms = []    
        for kvec in self.kvecs:
            argmat = np.zeros((N,N))
            for (m,n) in combinations(np.arange(N),2):
                argmat[m,n] = kvec.dot(self.atoms[m].coords-self.atoms[n].coords)
            norm_2 = np.sum(\
                    np.cos(argmat[np.triu_indices(N, k=1)]) *\
                        sxxpsyy[np.triu_indices(N, k=1)] +\
                    np.sin(argmat[np.triu_indices(N, k=1)]) *\
                        sxymsyx[np.triu_indices(N, k=1)])
            self.norms.append(0.5*(norm_1+norm_2))
        self.norms = np.array(self.norms).flatten()
    else:
        self.norms = None
    self.norms = self.comm.bcast(self.norms, root=root) 
    
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
	  r_t[0] = odeint(lindblad_bbgky_pywrap, \
	    mth_atom.state[alpha][0], times, args=(self,), Dfun=None)
	  r_t[1] = odeint(lindblad_bbgky_pywrap, \
	    mth_atom.state[alpha][1], times, args=(self,), Dfun=None)
	  r_t[2] = odeint(lindblad_bbgky_pywrap, \
	    mth_atom.state[alpha][2], times, args=(self,), Dfun=None)
	  r_t[3] = odeint(lindblad_bbgky_pywrap, \
	    mth_atom.state[alpha][3], times, args=(self,), Dfun=None)      
	  #Update the final state
	  self.local_atoms[atom_count].state[alpha][0] = r_t[0][-1] 
	  self.local_atoms[atom_count].state[alpha][1] = r_t[1][-1] 
	  self.local_atoms[atom_count].state[alpha][2] = r_t[2][-1] 
	  self.local_atoms[atom_count].state[alpha][3] = r_t[3][-1] 
	  for kcount in xrange(self.kvecs.shape[0]):
	    self.kvec = self.kvecs[kcount]
	    corrs_summedover_alpha[kcount] += \
	      self.field_correlations(alpha, r_t, mth_atom)
          if self.verbose and pbar_avail and self.comm.rank == root:
              bar.update(bar_pos)
              bar_pos += 1
          localdata[kcount][atom_count] = corrs_summedover_alpha[kcount]
            
	    
      duplicate_comm = Intracomm(self.comm)
      alldata = np.array([None for i in self.kvecs])
      for kcount in xrange(self.kvecs.shape[0]):
          localsum_data = np.sum(np.array(localdata[kcount]), axis=0)
          if self.comm.size == 1:
              alldata[kcount] = localsum_data
          else:
              alldata[kcount] = duplicate_comm.reduce(localsum_data, root=root)
          alldata[kcount] = alldata[kcount]/self.norms[kcount]
    
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
    else:
		self.steady_state = None
    #First disconnect, then broadcast
    if self.comm.rank == root:
        self.disconnect(self.steady_state)
    
    self.steady_state = self.comm.bcast(self.steady_state, root=root) 

    #Set the initial conditions and the reference states
    for (alpha, mth_atom) in product(np.arange(nalphas), self.local_atoms):  
      m = mth_atom.index
      #This is gonna have 4 states each as per method 3 in Lorenzo's writeup
      mth_atom.state[alpha] = [None, None, None, None]
      #Eq 64 in lorenzo's writeup for a = xyz or 012
      mth_atom.state[alpha][0] = np.copy(self.steady_state)
      self.tilde_trans(mth_atom.state[alpha][0],0,m)
      self.traceout_1p(mth_atom.state[alpha][0], m, alpha)
      self.traceout_2p(mth_atom.state[alpha][0], m, alpha)
      self.reconnect(mth_atom.state[alpha][0])
      
      mth_atom.state[alpha][1] = np.copy(self.steady_state)
      self.tilde_trans(mth_atom.state[alpha][1],1,m)
      self.traceout_1p(mth_atom.state[alpha][1], m, alpha)
      self.traceout_2p(mth_atom.state[alpha][1], m, alpha)
      self.reconnect(mth_atom.state[alpha][1])
	  
      mth_atom.state[alpha][2] = np.copy(self.steady_state)
      self.tilde_trans(mth_atom.state[alpha][2],2,m)
      self.traceout_1p(mth_atom.state[alpha][2], m, alpha)
      self.traceout_2p(mth_atom.state[alpha][2], m, alpha)
      self.reconnect(mth_atom.state[alpha][2])
	  
	#Eq 65 in Lorenzo's writeup:
      mth_atom.state[alpha][3] = np.copy(self.steady_state)
      self.traceout_1p(mth_atom.state[alpha][3], m, alpha)
      self.traceout_2p(mth_atom.state[alpha][3], m, alpha)
      self.reconnect(mth_atom.state[alpha][3])
	        
      mth_atom.refstate[alpha] = rvecs[alpha]
    
    #Reconnect steady state, then re-broadcast 
    if self.comm.rank == root:
        self.reconnect(self.steady_state)
    self.steady_state = self.comm.bcast(self.steady_state, root=root) 	
  
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
