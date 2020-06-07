# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:14:13 2015

@author: salmon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.externals.joblib import Memory
from sklearn.linear_model import lasso_path, enet_path
from sklearn.datasets.mldata import fetch_mldata
from sklearn import datasets, utils
from sklearn.utils.sparsefuncs import inplace_column_scale

from intro_forward_backward_source  import soft_thresholding




from mpi4py import MPI      

anysource = MPI.ANY_SOURCE  
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

############################################################################
#            Loading and visualizing the data
############################################################################

dataset_name = 'liver-disorders'
data = fetch_mldata(dataset_name)
X = data.data
y = data.target
X = X.astype(float)
y = y.astype(float)
y[y==2]=-1

# standardize data
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[np.isnan(X)] = 0.



############################################################################
#            Dataset splitting for distribution settting
############################################################################
n, p = np.shape(X)
nb_slaves = 5
block_size = np.floor(float(n/nb_slaves))
for i in range(nb_slaves):

    ind = i * block_size
    if rank == i+1:
        X = X[ind : ind + block_size,:]
        y = y[ind : ind + block_size]

############################################################################
#            Forward / Backward algorithm
############################################################################


alpha = 0.05	# regularization parameter
gamma = 0.1	# stepsize
max_iter = 500	# maximal number of iteration


def forward_backward_mpi(X, y, p, alpha, gamma, max_iter):

    w = np.zeros(p)
    ntot = np.array([0.0]) 
    gradient_sum = np.zeros(p)
    n = np.array([0.0])

    if rank != 0:
        ## Initialization ----
        n = np.array([float(X.shape[0])]) # houra for the floats !!!!!

    # ici on calcule ntot, le nombre total de points
    # utile pour que le proc 0 puisse normaliser le gradient
    comm.Reduce(n, ntot, MPI.SUM, root=0)
    
    ## Main Loop ----
    for iter in range(max_iter):

        ###################
        # MODIFY HERE
        ###################

        # gradient computation
        gradient = np.zeros(p)

        # Update of variable w
        
        # broadcast variable w


    return w


comm.Barrier()     # Blocking instruction that an agent wait here until every agent reach the instruction
w = forward_backward_mpi(X, y, p, alpha, gamma, max_iter)

# le processus 0 appelle la fonction non distribuee
# et affiche les deux w afin de verifier qu'ils sont identiques
from intro_forward_backward_source  import forward_backward
if rank == 0:
    print w
    w, _, _= forward_backward(X, y, alpha, gamma, max_iter)
    print w