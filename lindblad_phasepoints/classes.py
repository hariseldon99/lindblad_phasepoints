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
			  amplitude=1.0, detuning=0.0, cloud_rad = 1.0):
      
      """
       Usage:
       p = ParamData(latsize=100, \ 
		      drv_amp=1.0, drv_freq=0.0)
		      
       All parameters (arguments) are optional.
       
       Parameters:
       latsize    =  The size of your lattice as an integer. This can be in 
		     any dimensions
       amplitude  =  The periodic (cosine) drive amplitude 
		     Defaults to 1.0.
       detuning   =  The periodic (cosine) drive frequency, i.e.
		     detuning between atomic levels and incident light.
		     Defaults to 0.0.
       cloud_rad  =  The radius of the gas cloud of atoms. Defaults to 1.0

       Return value: 
       An object that stores all the parameters above. 
       Note that the momentum (magnitude) of theincidentlight
       is scaled to unity and propagates in the z-direction
      """

      self.latsize = latsize
      self.drv_amp, self.drv_freq = amplitude, detuning
      self.cloud_rad = cloud_rad
      #Set the momentum to be unit magnitude in z direction
      self.kvec = np.array([0.0, 0.0, 1.0]) 
	  
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