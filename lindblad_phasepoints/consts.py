from __future__ import division, print_function
# Some constant objects
from numpy import eye, zeros, array
from pprint import pprint

#Progressbar widgets
try:
  from progressbar import Bar, ETA, Percentage
  pbar_avail = True
  widgets_rnd = ['Creating atoms: ', Percentage(), ' ', Bar(), ' ', ETA()]
  widgets_bbgky = ['BBGKY  dynamics (root only): ',\
    Percentage(), ' ', Bar(), ' ', ETA()]
except ImportError:
  pbar_avail = False
  widgets = None

default_seed = 8
threshold = 1e-6
root = 0
#This is the kronecker delta symbol for vector indices
deltaij = eye(3)
#This is the Levi-Civita symbol for vector indices
eijk = zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

#Number of phase points
nalphas = 8

#Phase point vectors
rvecs = array([[1., 1., 1.],
		  [1.,-1.,-1.],
		  [-1.,-1., 1.],
		  [-1., 1.,-1.],
		  [1., -1., 1.],
		  [1., 1.,-1.],
		  [-1., 1., 1.],
		  [-1.,-1., -1.]])

#Maximum number of points
bigsize = 100000

#Steady state init and final times
ss_init_time = 0.0
ss_final_time = 300.0
ss_nsteps = 10000
ss_chunksize = 5000
int_method = 'lsoda'

#Verbosity function
def verboseprint(verbosity, *args):
    if verbosity:
        for arg in args:
            pprint(arg)
        print(" ")

blacklisted_keys = ["deltamn"]
        
#Format a dictionary for verboseprint by removing blacklisted_keys
def vbformat(in_dict):
    formatted =  {key:value for key, value in in_dict.items() if key not in\
                                                         blacklisted_keys}
    return formatted