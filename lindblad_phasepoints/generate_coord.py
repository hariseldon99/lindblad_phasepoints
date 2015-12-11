#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
#Try to import progressbars if available
try:
    import progressbar
    pbar_avail = True
except ImportError:
    pbar_avail = False

from consts import *
from classes import *

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
  
  x = np.random.uniform(-max,max, size=bigsize)
  y = np.random.uniform(-max,max, size=bigsize)
  z = np.random.uniform(-max,max, size=bigsize)
 
  manypoints = np.vstack((x,y,z)).T
  manypoints = manypoints[norm(manypoints,axis=1)<=max]
  mp_size = manypoints.shape[0]
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
