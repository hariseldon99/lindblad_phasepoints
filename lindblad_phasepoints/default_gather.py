import numpy as np
from mpi4py import MPI

def fib1(sizes):
  fib = np.zeros(sizes.size)
  for i, val in np.ndenumerate(sizes):
    (i,) = i
    fib[i] = np.sum(sizes[0:i])
  return fib

def gather_to_root(mpcomm, data, root=0):
  """
  Facilitates MPI Gather of all spin data for the atoms distributed 
  in the MPI Communicator
  """
  datasize_loc = data.size
  (locatoms, nalphas, times, xyz) =  data.shape
  natoms = mpcomm.reduce(locatoms, op=MPI.SUM, root=root)
  
  if mpcomm.rank == root:  
    allsizes = np.zeros(mpcomm.size)
    distrib_atoms = np.zeros_like(allsizes)
  else:
    allsizes = None
    distrib_atoms = None

  allsizes = mpcomm.gather(datasize_loc,allsizes, root=0)
  distrib_atoms = mpcomm.gather(locatoms, distrib_atoms, root=0)
  if mpcomm.rank == root:
    alldisps = fib1(np.array(allsizes))	  
  else:
    alldisps = None
  alldisps = np.array(alldisps)
 
  mpcomm.Barrier()

  if mpcomm.rank == root:
    full_data_size = np.sum(allsizes)
    recvbuffer = np.zeros(full_data_size)
    allsizes = tuple(allsizes)
    alldisps = tuple(alldisps)
  else:
    recvbuffer = None
    
  mpcomm.Gatherv(data,[recvbuffer,allsizes,alldisps, MPI.DOUBLE])
  
  if mpcomm.rank == root:
    recvbuffer = recvbuffer.reshape(natoms, nalphas, times, xyz)

  return np.array(recvbuffer), np.array(distrib_atoms)
