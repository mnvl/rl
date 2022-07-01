
import numpy as np
import multiprocessing as mp

num_procs = 16

processes = []
connections = []

def worker(connection, rank):
    for i in range(1000):
        print("recv", i)
        message = connection.recv()
        print("recv", i, message.shape)

        reply = np.zeros(1024*1024, dtype=np.uint8)
        print("send", i)
        connection.send(reply)
        print("send", i, reply.shape)

for rank in range(1, num_procs):
    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=worker, args=(child_conn, rank))

    processes.append(process)
    connections.append(parent_conn)

    process.start()

for j in range(1000):
    print("bcast")
    message = np.zeros(10*1024*1024, dtype=np.uint8)
    for conn in connections:
        conn.send(message)
    print("bcast", message.shape)

    for i, conn in enumerate(connections):
        print("recv", i+1)
        reply = conn.recv()
        print("recv", i+1, reply.shape)

for p in processes:
    p.join()
