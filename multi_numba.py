#!/usr/bin/env python

from mpi4py import MPI
import socket
import cluster_numba
import warnings

warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD

atoms = cluster_numba.make_cluster(int(1e4), 400)

start = MPI.Wtime()
energy = cluster_numba.potential(atoms)
end = MPI.Wtime()

print('I am {} on {}'.format(comm.rank, socket.gethostname()))
print('  time: {:.2f} s'.format(end-start))
