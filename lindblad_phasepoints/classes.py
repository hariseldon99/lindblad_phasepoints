#Class Library
from mpi4py import MPI
import numpy as np
from itertools import starmap
import operator as op
from consts import *
import math
from scipy.sparse import dia_matrix

class ParamData:
    """Class that stores Hamiltonian and lattice parameters 
       to be used in each dTWA instance. This class has no 
       methods other than the constructor.
    """
    
    def __init__(self, hopmat = None, latsize=11, \
			  drv_amp=1.0, drv_freq=0.0, cloud_rad = 1.0):
      
      """
       Usage:
       p = ParamData(hopmat = None, latsize=100, \ 
		      drv_amp=1.0, drv_freq=0.0)
		      
       All parameters (arguments) are optional.
       
       Parameters:
       hopmat 	=  The hopping matrix J for the Ising part of the 
		   Hamiltonian, i.e. J_{ij} \sigma^{xyz}_i \sigma^{xyz}_j
		   Example: In the 1-dimensional ising model with nearest 
		   neighbour hopping (open boundary conditions), J can be 
		   obtained via numpy by:
		   
		     import numpy as np
		     J = np.diagflat(np.ones(10),k=1) +\ 
			    np.diagflat(np.ones(10),k=-1)
		   
		   The diagonal is expected to be 0.0. There are no 
		   internal checks for this. When set to 'None', the module
		   defaults to the hopmat of the 1D ising model with long 
		   range coulomb hopping and open boundary conditions.	   
       latsize  =  The size of your lattice as an integer. This can be in 
		   any dimensions
       drv_amp =   The periodic (cosine) drive amplitude 
		   Defaults to 1.0.
       drv_freq =  The periodic (cosine) drive frequency 
		   Defaults to 0.0.
       cloud_rad = The radius of the gas cloud of atoms. Defaults to 1.0	   
			   
       Return value: 
       An object that stores all the parameters above. 
      """

      self.latsize = latsize
      self.drv_amp, self.drv_freq = drv_amp, drv_freq
      N = self.latsize
      self.fullsize_2ndorder = 3 * N + 9 * N**2
      self.deltamn = np.eye(N)
      if(hopmat == None): #Use the default hopping matrix
	#This is the dense Jmn hopping matrix with inverse 
	#power law decay for periodic boundary conditions.
	J = dia_matrix((N, N))
	mid_diag = np.floor(N/2).astype(int)
	for i in xrange(1,mid_diag+1):
	  elem = pow(i, -1.0)
	  J.setdiag(elem, k=i)
	  J.setdiag(elem, k=-i)
	for i in xrange(mid_diag+1, N):
	  elem = pow(N-i, -1.0)
	  J.setdiag(elem, k=i)
	  J.setdiag(elem, k=-i)
	  self.jmat = J.toarray()
      else: #Take the provided hopping matrix
	  self.jmat = hopmat  
	  
class Atom:
  """
  Class that stores all the data specifying a single atom 
  in the cloud (index and coordinates). 
  Also has methods for extracting the data and calculating 
  distance between 2 atoms
  """
  
  def __init__(self, coords = np.array([0.0,0.0]), index = 0):
      """
       Usage:
       import numpy as np
       c = np.array([1.2,0.4])
       a = Atom(coords = c, index = 3)
       
       All arguments are optional.
       
       Parameters:
       coords 	=  The 2D coordinates of the atom in the cloud, entered as a 
		   numpy array of double precision floats np.array([x,y])
		   
       index 	=  The index of the atom while being counted among others.
		   These are counted from 0
       latsize  =  The size of your lattice as an integer. This can be in 
		   any dimensions
			   
       Return value: 
       An atom object  
      """
      
      if(coords.size == 2):
	self.index = index
	self.coords = coords
      else:
	raise ValueError('Incorrect 2D coordinates %d' % (coords))
    
  def extract(self):
    """
    Usage:
    a = atom(coords = (1.2,0.4), index = 3)
    print a.extract()
    
    Returns a tuple containing the atom's count index and
    the coordinates tuple.
    """
    return (self.index, self.coords)
  
  def distance_to(self, other_atom):
    """
    Returns the distance between 2 atoms
    """
    sc, oc = self.coords, other_atom.coords
    return math.hypot((sc-oc)[0], (sc-oc)[1])