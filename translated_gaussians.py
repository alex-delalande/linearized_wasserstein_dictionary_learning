#!/usr/bin/env python
# coding: utf-8

# # Dictionary learning on translated gaussians
# 
# Here we observe a dataset of 5 translated univariate gaussians and learn a dictionary of 2 atoms compressing them. The atoms are imposed to lie in the convex envelop of the data and the codes are imposed to be in the simplex.
# 
# Imports:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from utils import *
from dictionary_learning import *


# ## Dataset

# Create dataset:

# In[2]:


b = 100
b_quantiles = 200
m = 2000
n = 5
width = 0.2
x_range = (-4, 16)

samples = np.zeros((m, n))
Histograms = np.zeros((b, n))
np.random.seed(6)

for i in range(n):
    s = np.random.normal(2*(i+1), size=m)
    samples[:, i] = s
    Histograms[:, i] = np.histogram(s, b, range=x_range)[0]/m


# Convert observed histograms into quantiles and observe dataset:

# In[3]:


estimated_Quantiles = histograms_to_quantiles(Histograms, b_quantiles=b_quantiles, quantile_range=x_range)

plt.figure(figsize=(18, 7))
x_axis = np.linspace(x_range[0], x_range[1], b)
x_axis_quantiles = np.linspace(0, 1, b_quantiles)

for i in range(5):
    idx = i
    plt.subplot(2, 5, i+1)
    plt.bar(x_axis, Histograms[:, idx], width=width)
    plt.tight_layout(pad=3.0)
    plt.xlim((-3, 15))
    plt.ylim((0, 0.1))
    if (i%5==2): plt.title('Histogram representation' + '\n' + r'$\mu_{}$'.format(i+1), fontdict={'fontsize': 20})
    else: plt.title(r'$\mu_{}$'.format(i+1), fontdict={'fontsize': 20})
    plt.subplot(2, 5, i+6)
    plt.plot(x_axis_quantiles, estimated_Quantiles[:, idx])
    plt.xlim((-0.05, 1.05))
    plt.ylim((-2, 13))
    if (i%5==2): plt.title('Quantile representation' + '\n' + r'$\mu_{}$'.format(i+1), fontdict={'fontsize': 20})
    else: plt.title(r'$\mu_{}$'.format(i+1), fontdict={'fontsize': 20})
plt.savefig("outputs/input_gaussians.pdf", format="pdf", bbox_inches="tight")


# ## Dictionary learning in quantile domain

# Learn a dictionary of 2 atoms and observe atoms in the quantile domain:

# In[4]:


niter = 100
step = 5e-5
DL = DL_block_coordinate_descent(n, k=2)
DL.fit(estimated_Quantiles, niter, codes_in_simplex=True, atoms_in_data=True, step=step)
D = DL.D
Lambda = DL.Lambda

plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(niter-10), DL.errors[10:])
plt.subplot(1, 2, 2)
plot_atoms(D.T, width/40, b_quantiles, title="Dictionary atoms - Quantile domain",
           x_range=(0, 1))


# Atoms in the histogram domain:

# In[5]:


plot_atoms(quantiles_to_histograms(D, b, x_range).T, width, b,
           title="Histograms of dictionary atoms learnt from quantile representations",
           x_range=x_range)
plt.savefig("outputs/atoms_quantiles.pdf", format="pdf", bbox_inches="tight")


# Observe reconstructions:

# In[6]:


rec = quantiles_to_histograms(D.dot(Lambda), b, quantile_range=x_range) 

plt.figure(figsize=(18, 3))
for i in range(n):
    plt.subplot(1, n, i+1)
    plt.bar(x_axis, Histograms[:, i], color='g', label='Original', alpha=0.3, width=width)
    plt.bar(x_axis, rec[:, i], color='r', label='Reconstruction', alpha=0.3, width=width)
    plt.title(r'$\mu_{}$'.format(i+1))
    plt.legend()
plt.savefig("outputs/reconstruction_quantiles.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Observe weights used to build the atoms and codes used for the reconstructions:

# In[7]:


print("Weights for the atoms:\n", DL.W)

print("\nWeights for the reconstructions:\n", DL.Lambda.T)


# ## Dictionary learning in histogram domain

# Learn a dictionary of 2 atoms and observe atoms in the histogram domain:

# In[8]:


niter = 200
step = 5e-1
DL = DL_block_coordinate_descent(n, k=2)
DL.fit(Histograms, niter, codes_in_simplex=True, atoms_in_data=True, step=step)
D = DL.D
Lambda = DL.Lambda


plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(niter-10), DL.errors[10:])
plt.subplot(1, 2, 2)
plot_atoms(D.T, width, b, title="Dictionary atoms - Histogram domain", x_range=x_range)


# Observe only atoms:

# In[9]:


plot_atoms(D.T, width, b,
           title="Histograms of dictionary atoms learnt from histograms representations",
           x_range=x_range)
plt.savefig("outputs/atoms_histograms.pdf", format="pdf", bbox_inches="tight")


# Observe reconstructions:

# In[10]:


rec = D.dot(Lambda)

plt.figure(figsize=(18, 3))
for i in range(n):
    plt.subplot(1, n, i+1)
    plt.bar(x_axis, Histograms[:, i], color='g', label='Original', alpha=0.3, width=width)
    plt.bar(x_axis, rec[:, i], color='r', label='Reconstruction', alpha=0.3, width=width)
    plt.title(r'$\mu_{}$'.format(i+1))
    plt.legend()
plt.savefig("outputs/reconstruction_histograms.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Observe weights used to build the atoms and codes used for the reconstructions:

# In[11]:


print("Weights for the atoms:\n", DL.W)

print("\nWeights for the reconstructions:\n", DL.Lambda.T)


# In[ ]:




