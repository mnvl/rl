
import os
import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("process", rank, size)

message_size = 16*1024
reply_size = 1024*1024

for i in range(1000):
    if rank == 0:
        message = np.zeros(message_size, dtype=np.uint8)
        print("bcast", rank, message.shape)
        comm.bcast(message, root=0)

        time.sleep(100.0)

        for j in range(1, size):
            print("recv", rank)
            reply = comm.recv(source=j)
            print("recv", rank, reply.shape)
    else:
        print("bcast", rank)
        message = comm.bcast(None, root=0)
        print("bcast", rank, message.shape)

        print("send", rank)
        reply = np.zeros(reply_size, dtype=np.uint8)
        comm.send(reply, dest=0)
        print("send", rank, reply.shape)
