# Some constant objects
from numpy import eye, zeros, array

#Progressbar widgets
try:
  from progressbar import Bar, Counter, ETA, Percentage
  pbar_avail = True
  widgets_rnd = ['Creating atoms: ', Percentage(), ' ', Bar(), ' ', ETA()]
  widgets_bbgky = ['BBGKY  dynamics (root only): ',\
    Percentage(), ' ', Bar(), ' ', ETA()]
except ImportError:
  pbar_avail = False
  widgets = None

seed = 8
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