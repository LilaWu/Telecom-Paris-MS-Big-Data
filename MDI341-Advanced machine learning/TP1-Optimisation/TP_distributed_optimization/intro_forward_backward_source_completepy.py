
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:14:13 2015

@author: bellet, jelassi, salmon
"""

import numpy as np
import matplotlib.pyplot as plt

############################################################################
#            Loading and visualizing the data
############################################################################

def plot_2d(X, y):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""
    plt.figure()
    symlist = ['o', 's', '*', 'x', 'D', '+', 'p', 'v', 'H', '^']
    collist = ['blue', 'red', 'purple', 'orange', 'salmon', 'black', 'grey',
               'fuchsia']

    labs = np.unique(y)
    idxbyclass = [y == labs[i] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.plot(X[idxbyclass[i], 0], X[idxbyclass[i], 1], '+',
                 color=collist[i % len(collist)], ls='None',
                 marker=symlist[i % len(symlist)])
    plt.ylim([np.min(X[:, 1]), np.max(X[:, 1])])
    plt.xlim([np.min(X[:, 0]), np.max(X[:, 0])])
    plt.show()



############################################################################
#            SOFT THRESHOLDING FUNCTION
############################################################################

# Soft thresholding function
def soft_thresholding(w, l):
    dim = w.size
    ws = np.zeros(dim)
    for i in range(dim):
        ###################
        # MODIFY HERE
        ###################
        if w[i] > l:
            ws[i] = w[i] - l
        elif w[i] < -l:
            ws[i] = w[i] + l
        else:
            ws[i] = 0.0

        ###########################
        # DO NOT MODIFY AFTER HERE
        ###########################
    return ws




def forward_backward(X, y, alpha, gamma, max_iter):

    ## Initialization ----
    [n, p] = np.shape(X)
    w = np.zeros(p)
    obj_fun = np.zeros(max_iter)
    sparsity = np.zeros(max_iter)

    ## Main Loop ----
    for iter in range(max_iter):

        ## -------------- Update START -------------
        ###################
        # MODIFY HERE
        ###################

        # gradient computation
        gradient = np.zeros(p)
        for i in range(n):
            gradient -=  y[i] * X[i,:] / (1+np.exp(y[i] * np.dot(w,X[i,:])))
        gradient = (1.0 / n) * gradient

        # Update of variable w
        w = soft_thresholding(w - gamma * gradient, alpha * gamma)

        # objective function computation
        for i in range(n):
            obj_fun[iter] += np.log(1+np.exp(-y[i] * np.dot(w,X[i,:])))
        obj_fun[iter] =  (1.0 / n) * obj_fun[iter] + alpha * np.sum(np.abs(w))
        
        # sparsity computation
        sparsity[iter] = sum(w!=0)

    return w, obj_fun, sparsity






def sub_gradient_descent(X, y, alpha, gamma, max_iter):

    ## Initialization ----
    [n, p] = np.shape(X)
    w = np.zeros(p)
    obj_fun = np.zeros(max_iter)
    sparsity = np.zeros(max_iter)

    ## Main Loop ----
    for iter in range(max_iter):

        ###################
        # MODIFY HERE
        ###################

        # gradient computation
        gradient = np.zeros(p)
        for i in range(n):
            gradient -=  y[i] * X[i,:] / (1+np.exp(y[i] * np.dot(w,X[i,:])))
        gradient = (1.0 / n) * gradient

        # subgradient computation
        sub_gradient = gradient + alpha * np.sign(w)
        # update parameter
        w = w - gamma * sub_gradient

        # objective function computation
        for i in range(n):
            obj_fun[iter] += np.log(1+np.exp(-y[i] * np.dot(w,X[i,:])))
        obj_fun[iter] =  (1.0 / n) * obj_fun[iter] + alpha * np.sum(np.abs(w))
        
        # sparsity computation
        sparsity[iter] = sum(w!=0)

    return w, obj_fun, sparsity


