import numpy as np

def gather_to_root(mpcomm, datatype, data, root=0):
  datasize_loc = data.size
  datadisp_loc = mpcomm.rank

  if mpcomm.rank == root:
    allsizes = np.zeros(mpcomm.size)
  else:
    allsizes = None

  allsizes = mpcomm.gather(datasize_loc,allsizes, root=0)

  if mpcomm.rank == root:
    alldisps = np.zeros(mpcomm.size)
    for n in xrange(mpcomm.size):
      alldisps[n] = alldisps[n-1] + allsizes[n] - 1
  else:
    alldisps = None

  mpcomm.Barrier()

  if mpcomm.rank == root:
    full_data_size = np.sum(allsizes)
    recvbuffer = np.zeros(full_data_size)
    allsizes = tuple(allsizes)
    alldisps = tuple(alldisps)
  else:
    recvbuffer = None
    
  mpcomm.Gatherv(data,[recvbuffer,allsizes,alldisps, datatype])

  return recvbuffer