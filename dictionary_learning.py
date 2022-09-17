# Linearized Wasserstein Dictionary Learning
# September 14, 2022


# imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from utils import euclidean_proj_simplex


class DL_block_coordinate_descent:
    ''' Unconstrained dictionary learning with block coordinate descent.'''
    
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.Lambda = np.random.rand(k, n)
        self.errors = []
        
    def fit(self, X, niter=500, codes_in_simplex=False, atoms_in_data=False, step=1e-6):
        # initialization depending on chosen algorithm
        if codes_in_simplex:
            self.Lambda = np.apply_along_axis(euclidean_proj_simplex, 0, self.Lambda)
        if atoms_in_data:
            self.W = np.random.rand(self.n, self.k)
            self.W = np.apply_along_axis(euclidean_proj_simplex, 0, self.W)
        # coordinate descent
        for t in range(niter):
            # update D
            if atoms_in_data:
                self.W -= step * 2*(X.T).dot(X.dot(self.W).dot(self.Lambda) - X).dot(self.Lambda.T)
                #self.W = (self.Lambda.T).dot( np.linalg.pinv(self.Lambda.dot(self.Lambda.T)) )
                self.W = np.apply_along_axis(euclidean_proj_simplex, 0, self.W)
                self.D = X.dot(self.W)
            else:
                inv_Lambda = np.linalg.pinv(self.Lambda.dot(self.Lambda.T))
                self.D = (X.dot(self.Lambda.T)).dot(inv_Lambda)
            # update lambda
            inv_D = np.linalg.pinv((self.D.T).dot(self.D))
            self.Lambda = inv_D.dot((self.D.T).dot(X))
            if codes_in_simplex:
                self.Lambda = np.apply_along_axis(euclidean_proj_simplex, 0, self.Lambda)
            # compute error
            rec = self.D.dot(self.Lambda)
            self.errors.append(mean_squared_error(X.T, rec.T))          

