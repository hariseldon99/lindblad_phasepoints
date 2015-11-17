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
    
    def __init__(self, latsize=11, \
			  drv_amp=1.0, drv_freq=0.0, cloud_rad = 1.0,\
			    kvec_theta=0.0, kvec_phi=0.0):
      
      """
       Usage:
       p = ParamData(latsize=100, \ 
		      drv_amp=1.0, drv_freq=0.0)
		      
       All parameters (arguments) are optional.
       
       Parameters:
       latsize    =  The size of your lattice as an integer. This can be in 
		     any dimensions
       drv_amp    =   The periodic (cosine) drive amplitude 
		      Defaults to 1.0.
       drv_freq   =  The periodic (cosine) drive frequency 
		     Defaults to 0.0.
       cloud_rad  =  The radius of the gas cloud of atoms. Defaults to 1.0
       kvec_theta =  Azimuthal angle of incident laser beam
                     Note that the momentum (magnitude) is scaled to unity
       kvec_phi   =  Polar angle of incident laser beam
		     Note that the momentum (magnitude) is scaled to unity

       Return value: 
       An object that stores all the parameters above. 
      """

      self.latsize = latsize
      self.drv_amp, self.drv_freq = drv_amp, drv_freq
      self.cloud_rad = cloud_rad
      kx = np.sin(kvec_theta) * np.cos(kvec_phi)
      ky = np.sin(kvec_theta) * np.sin(kvec_phi)
      kz = np.cos(kvec_theta)
      self.kvec = np.array([kx,ky,kz])
      N = self.latsize
      self.fullsize_2ndorder = 3 * N + 9 * N**2
      self.deltamn = np.eye(N)
 
	  
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
      if(coords.size == 3):
	self.index = index
	self.coords = coords
      else:
	raise ValueError('Incorrect 3D coordinates %d' % (coords))
    
  def extract(self):
    """
    Usage:
    a = atom(coords = (1.2,0.4, 0.5), index = 3)
    print a.extract()
    
    Returns a tuple containing the atom's count index and
    the coordinates tuple.
    """
    return (self.index, self.coords)
  
  def distance_to(self, other_atom):
    """
    Returns the distance between 2 atoms
    """
    (x1,y1,z1), (x2,y2,z2) = self.coords, other_atom.coords 
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
  
  
