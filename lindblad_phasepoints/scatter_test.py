from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num=14

sendbuf = None
if rank == 0:
    sendbuf = np.random.random(num)
    sendbuf = np.array_split(sendbuf, size)
    locsize = np.array([s.size for s in sendbuf])
    print "sendbuf = ", sendbuf
    #print "sendsizes = ",locsize
else:
    locsize = None
locsize = comm.scatter(locsize, root=0)
recvbuf = np.empty(locsize, dtype="float64")
recvbuf = comm.scatter(sendbuf, root=0)
print comm.rank, "," ,recvbuf
