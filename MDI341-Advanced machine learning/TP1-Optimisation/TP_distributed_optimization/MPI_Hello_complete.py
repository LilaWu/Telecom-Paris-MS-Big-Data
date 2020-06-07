#######################################################
# File: 	MPI_Hello.py
# Info:		A simple MPI Hello World
# Input:	To call in shell as 'mpiexec -n 5 python MPI_Example.py'
#######################################################

##------------------------------------------------------------------
## Packages Import
##------------------------------------------------------------------

from mpi4py import MPI		# We import the the module MPI of the package mpi4py
import numpy as np
##------------------------------------------------------------------
## Global MPI Variables
##------------------------------------------------------------------

anysource = MPI.ANY_SOURCE 	# New name for MPI Global variable saying one can receive from anysource (see below)
comm = MPI.COMM_WORLD		# New name for MPI Global Environment
size = comm.Get_size()		# The size of the world i.e. the number of active agents
rank = comm.Get_rank()		# MY agent number (this variable DIFFERS from one agent to another)

##------------------------------------------------------------------
##  Work after here
##------------------------------------------------------------------

print size, rank

comm.Barrier()     # Blocking instruction that an agent wait here until every agent reach the instruction

if rank == 1:
	comm.send("hello world", dest = 3)
	print "%d is sending a message"%rank

if rank == 3:
	msg = comm.recv(source = 1)
	print "%d is receiving the message: %s"%(rank,msg)

comm.Barrier()     # Blocking instruction that an agent wait here until every agent reach the instruction

n = 5
x = np.random.rand(n)
print x


#initialize sums
output = np.zeros(n)
#compute distributed sums
comm.Reduce(x, output, MPI.SUM, root = 0)

if rank == 0:
	print "the sum is"
	print output


#brodacast all
comm.Bcast(output, root = 0)

if rank != 0:
	print output
