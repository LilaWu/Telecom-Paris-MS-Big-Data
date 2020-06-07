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

