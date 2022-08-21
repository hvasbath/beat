#!/usr/bin/env python
"""
Parallel Tempering algorithm - TOYEXAMPLE with mpi4py
"""
from mpi4py import MPI
from numpy import random

nsamples = 100


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)


def metrop_select(m1, m2):
    u = random.rand()
    if u < 0.5:
        print("Rejected swap")
        return m1, m2
    else:
        print("Accepted swap")
        return m2, m1


def master_process(comm, size, tags, status):

    num_workers = size - 1
    tasks = range(num_workers)
    chain = []
    active_workers = 0
    # start sampling of chains with given seed
    print("Master starting with %d workers" % num_workers)
    for i in range(num_workers):
        comm.recv(source=MPI.ANY_SOURCE, tag=tags.READY, status=status)
        source = status.Get_source()
        comm.send(tasks[i], dest=source, tag=tags.START)
        print("Sent task to worker %i" % source)
        active_workers += 1

    print("Parallel tempering ...")
    print("----------------------")

    while True:
        m1 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source1 = status.Get_source()
        print("Got sample 1 from worker %i" % source1)
        m2 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source2 = status.Get_source()
        print("Got sample 2 from worker %i" % source1)

        m1, m2 = metrop_select(m1, m2)
        print("samples 1, 2 %i %i" % (m1, m2))
        chain.extend([m1, m2])
        if len(chain) < nsamples:
            print("Sending states back to workers ...")
            comm.send(m1, dest=source1, tag=tags.START)
            comm.send(m2, dest=source2, tag=tags.START)
        else:
            print("Requested number of samples reached!")
            break

    print("Master finishing, recorded chain:")
    print(chain)
    print("Closing ...")
    for i in range(1, size):
        print("sending signal to close to %i" % i)
        comm.send(None, dest=i, tag=tags.EXIT)

        print("Closed worker %i" % i)
        active_workers -= 1


def worker_process(comm, rank, tags, status):
    # Worker processes execute code below
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    comm.send(None, dest=0, tag=tags.READY)

    while True:
        print("Receiving ...")
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        print("received!")

        tag = status.Get_tag()

        if tag == tags.START:
            # Do the work here
            result = task + 1
            print("attempting to send ...")
            comm.send(result, dest=0, tag=tags.DONE)
            print("sending worked ...")
        elif tag == tags.EXIT:
            print("went through exit")
            break


def pt_sample():
    # Define MPI message tags
    tags = enum("READY", "DONE", "EXIT", "START")

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    size = comm.size  # total number of processes
    rank = comm.rank  # rank of this process
    status = MPI.Status()  # get MPI status object

    if rank == 0:
        print("Here")
        master_process(comm, size, tags, status)
    else:
        print("worker")
        worker_process(comm, rank, tags, status)

    print("Done!")


if __name__ == "__main__":
    print("here main")
    pt_sample()
