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


import time
import numpy as np
from scipy import linalg, io, sparse
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Memory
from sklearn.linear_model import lasso_path, enet_path
from sklearn.datasets.mldata import fetch_mldata
from sklearn import datasets, utils
from sklearn.utils.sparsefuncs import inplace_column_scale

from intro_forward_backward_source_correction import *

plt.close('all')

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

[n, p] = np.shape(X)

# Visualisation de la premiere variable
eps=0.2*np.random.rand(n)
plt.figure()
plt.plot(X[:,4],y+eps,'.')
plt.ylim([-2,2])
plt.show()

############################################################################
#            SOFT THRESHOLDING FUNCTION
############################################################################


plt.figure()
grid = np.arange(-1,1,0.1)
plt.plot(grid,soft_thresholding(grid,0.4))
plt.title("Seuillage doux")
plt.show()



############################################################################
#            Forward / Backward algorithm
############################################################################


alpha = 0.05	# regularization parameter
gamma = 0.1	# stepsize
max_iter = 500	# maximal number of iteration



w, obj_fun, sparsity = forward_backward(X, y, alpha, gamma, max_iter)

plt.figure()
plt.plot(obj_fun)
plt.title("objectif")
plt.show()

plt.figure()
plt.plot(sparsity)
plt.title("Pseudo norm l0")
plt.show()


# BEWARE to alpha*n with sklearn !!!
clf = LogisticRegression(fit_intercept=False, penalty='l1', C = 1 / (alpha * n))
clf.fit(X,y)
print clf.coef_
print w


