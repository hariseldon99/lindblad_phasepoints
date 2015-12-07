#Class Library
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
from itertools import starmap
import operator as op
from consts import *
import math
from scipy.sparse import dia_matrix
from scipy.spatial.distance import pdist

#Try to import progressbars if available
try:
    import progressbar
    pbar_avail = True
except ImportError:
    pbar_avail = False

def generate_coordinates(size, min = 0.0, max = 1.0, verbose=False):
  """
  Usage: 
  a = generate_coordinates(size, min = 0.1, max=3.0)
  
  Parameters:
  size 		  = size of output
  min (optional)  = minimum 2-norm between any 2 
			    elements of output. Defaults to 0
  verbose 	  = Boolean. If set to True and the 
			    progressbar2 module is installed,
			    then shows a progresbar. Default is 
			    False

 Returns:
  Tuple (array, mdist)
  array = A numpy array of shape (size,3) of random elements. Each element
	      is a 3-dimensional cartesian vector whose norm is less than 'max' 
	      (optional, default is 1.0). This one uses a smart method
	      1. Make a random list of many many points  
	      2. Pick a point from this list at random
	      3. Remove from list all points that lie within 'min' distance of it
	      4. Repeat above steps 'size' times and return the picked points
 mdist = Minimum distance between 2 elements in array	      
  """  
  if pbar_avail and verbose:
      bar = progressbar.ProgressBar(widgets=widgets_rnd,\
              max_value=size-1, redirect_stdout=False)
  np.random.seed(seed)
  r = np.linspace(0.0,max, num=bigsize)
  theta = np.random.uniform(0.0, np.pi, size=bigsize)
  phi = np.random.uniform(0.0, 2.0 * np.pi, size=bigsize)
  manypoints = np.vstack((r * np.sin(theta) * np.cos(phi),\
    r * np.sin(theta) * np.sin(phi), r * np.cos(theta))).T
  mp_size = bigsize
  points = []
  atom_count = 0
  while atom_count < size:
    if pbar_avail and verbose:
      bar.update(atom_count)    
    p =  manypoints[np.random.randint(mp_size),:]
    manypoints = manypoints[norm(p - manypoints, axis=1) > min]
    mp_size = manypoints[:,0].size
    points.append(p)
    atom_count += 1
  return points, np.amin(pdist(np.array(points), 'euclidean'))

class ParamData:
    """Class that stores Hamiltonian and lattice parameters 
       to be used in each dTWA instance. This class has no 
       methods other than the constructor.
    """
    
    def __init__(self, latsize=11, amplitude=1.0, detuning=0.0, \
      cloud_rad=100.0, theta=0.0, mtime = 0.0):
      
      """
       Usage:
       p = ParamData(latsize=100, \ 
		      amplitude=1.0, detuning=0.0)
		      
       All parameters (arguments) are optional.
       
       Parameters:
       latsize   	 =  The size of your lattice as an integer. This can be in 
			      any dimensions
       amplitude  =  The periodic (cosine) drive amplitude 
			      Defaults to 1.0.
       detuning   	 =  The periodic (cosine) drive frequency, i.e.
			      detuning between atomic levels and incident light.
			      Defaults to 0.0.
       cloud_rad  =  The radius of the gas cloud of atoms. Defaults to 100.0
       theta	 	 =  Azimuthal angle of the incident laser beam. Defaults to 0
       mtime	 =  Time at which the correlations are evaluated i.e. the 
			      quantity <E^\dagger (mtime) * E(mtime+t)> where
			      E is the electric field. Defaults to 0 ie initial correlations.

       Return value: 
       An object that stores all the parameters above. 
       Note that the momentum (magnitude) of theincident light
       is scaled to unity and propagates in the z-direction
       by default unless you set the azimuth theta to a specific value
      """

      self.latsize = latsize
      self.drv_amp, self.drv_freq = amplitude, detuning
      self.cloud_rad = cloud_rad
      #Set the momentum to be unit magnitude in z direction
      self.kvec = np.array([np.sin(theta), 0.0, np.cos(theta)]) 
      self.corr_norm = 16.0 * self.latsize
      self.cloud_density = \
	self.latsize/((4./3.) * np.pi * pow(self.cloud_rad,3.0))
      self.intpt_spacing = 1./pow(self.cloud_density,1./6.)
      self.mtime = mtime

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
