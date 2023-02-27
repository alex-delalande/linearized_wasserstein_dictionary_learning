#!/usr/bin/env python
# coding: utf-8

# # Dictionary learning & classification on the ORL 
# 
# Comparison with [Fast Dictionary Learning with a Smoothed Wasserstein Loss (Rolet, Cuturi, Peyré)](http://marcocuturi.net/Papers/rolet16fast.pdf) and [Nonnegative Matrix Factorization with Earth Mover’s Distance Metric (Sandler, Lindenbaum)](https://d1wqtxts1xzle7.cloudfront.net/53866503/SandlerL_09.pdf?1500139054=&response-content-disposition=inline%3B+filename%3DNonnegative_Matrix_Factorization_with_Ea.pdf&Expires=1593727623&Signature=HFuLzAB1AaJ4Iv2NBArPhSGlLPwZGTCMEYE-wP2K4mxTvHtLxqIQrsJmM7fGuMczBI6c~7LfDrmBscNZWdPn-73yh24kc~l1ECOMPftgxiGvsgaKIl0wVQeOY4vBVFWt5fissrVdVyVhxi-yXa3WwRI9u6H2gudUiJ~gOUTeZqsuJCc8V-mXu2OWWc12SThvsbTb79Wfrzb1vE7sIBntJfN50n2mp~IH~LmuqDAUjwtezFZdGByFMLJSK1rpVAfuE~uvREa4w-Auhsh3uWrwHriUnz4wVpsHPqKJfMaglXxaFiDk4qfZ5djfFmu26eDWstNxrGPZ1bpzWEhhAodJHg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA).

# In[1]:


import numpy as np
import seaborn
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.image as img
import skimage.measure
from sklearn.metrics import pairwise_distances

from utils import *
from dictionary_learning import *

import optimal_transport


# ## Import images:

# Import images and preprocess (downsample with mean pooling):

# In[2]:


x_length, y_length = 112, 92
n, p = 400, 10
downsampling_factor = 3

x_length = x_length//downsampling_factor
y_length = y_length//downsampling_factor

Images = np.zeros((n, x_length, y_length))
y = np.zeros(n)

for i in range(n//p):
    for j in range(p):
        y[i*10 + j] = i
        image = read_pgm("orl_dataset/s{}/{}.pgm".format(i+1, j+1), byteorder='<')
        image = skimage.measure.block_reduce(image, (downsampling_factor, downsampling_factor), np.mean)
        Images[i*10 + j] = image[:-1, :-1]
        
y = y.astype(int)


# Show random images and corresponding labels:

# In[3]:


plt.figure(figsize=(20, 5))
for i in range(10):
    idx = np.random.choice(n)
    image = Images[idx]
    plt.subplot(1, 10, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(y[idx])
    plt.axis('off')
plt.savefig("outputs/orl_examples.pdf", bbox_inches="tight")


# Distance matrix on the natural images:

# In[4]:


orl_natural_images = Images.reshape((n, -1)).T
distance_matrix_natural_images = pairwise_distances(orl_natural_images.T)

from_, to_ = 0, 400

plt.figure(figsize=(11, 10))
seaborn.heatmap(distance_matrix_natural_images[from_:to_, from_:to_])
plt.show()


# ## Compute potentials/transport plans:

# Convert images into point clouds (observe some):

# In[5]:


point_clouds, masses = images_to_point_clouds(Images)

plt.figure(figsize=(20, 2))
cloud = point_clouds[0]
for i in range(10):
    idx = np.random.choice(n)
    plt.subplot(1, 10, i+1)
    plt.scatter(point_clouds[idx][:, 0], point_clouds[idx][:, 1], s=5e3*masses[idx])
    plt.title(idx)
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.axis('off')


# **Compute transport plans:** `Duration: 6.5s`.
# 
# Observe some push forwards from transport plans (note that the value of the grid size parameter does not allow to recover the original image).

# In[6]:


bins = x_length
grid_size = 50
start = time.time()
ot = optimal_transport.semi_discrete_ot(grid_size)
ot.fit_transport_plans(point_clouds, masses)
end = time.time()
print("Duration:", (end - start))
# save transport plans
np.save('orl_dataset/orl_transport_plan_grid_{}.npy'.format(grid_size), ot.transport_plans.T)

orl_transport_plan = ot.transport_plans.T
#orl_transport_plan = np.load('orl_transport_plan_grid_{}.npy'.format(grid_size))
print(orl_transport_plan.shape)

def get_push_forward(T, form='histogram', bins=x_length):
    # extract (push-forward) point cloud
    grid_size = np.int(np.sqrt(len(T)//2))
    point_cloud = np.reshape(T, (grid_size*grid_size, 2))
    # convert point cloud into histogram
    if form=='histogram':
        histogram_2D = np.histogram2d(point_cloud[:, 0], point_cloud[:, 1],
                                      bins=np.array([int(y_length/x_length * bins), bins]),
                                        range=np.array([[0, 1], [0, 1]]))
        return np.rot90(histogram_2D[0])/np.sum(histogram_2D[0])
    else:
        return point_cloud

plt.figure(figsize=(34, 4))
for i in range(5):
    idx = np.random.choice(n)
    image = Images[idx]
    plt.subplot(2, 10, i+1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(get_push_forward(orl_transport_plan[:, idx], bins=bins), cmap='gray')
    plt.title('Push forward')
    plt.axis('off')


# Distance matrix on the transport plans:

# In[7]:


distance_matrix_transport_plan = pairwise_distances(orl_transport_plan.T)

from_, to_ = 0, 400

plt.figure(figsize=(11, 10))
seaborn.heatmap(distance_matrix_transport_plan[from_:to_, from_:to_])
plt.show()


# ## Define experiment

# In[8]:


def orl_experiment(X, k, niter=20, n=n, p=p, verbose=True):
    
    # separate X into a train and a test sets
    training_idxes, testing_idxes = np.zeros(n//2), np.zeros(n//2)
    for i in range(n//p):
        possible_idxes = np.arange(i*p, (i+1)*p)
        sample = np.random.choice(possible_idxes, p//2, replace=False)
        training_idxes[(i*p)//2:((i+1)*p)//2] = sample
        testing_idxes[(i*p)//2:((i+1)*p)//2] = np.setdiff1d(possible_idxes, sample)
    training_idxes, testing_idxes = training_idxes.astype(int), testing_idxes.astype(int)
    X_train, y_train = np.take(X, training_idxes, axis=1), y[training_idxes]
    X_test, y_test = np.take(X, testing_idxes, axis=1), y[testing_idxes]
    
    # dictionary learning on the train set (atoms and codes)
    start = time.time()
    DL = DL_block_coordinate_descent(n//2, k)
    DL.fit(X_train, niter, codes_in_simplex=True)
    D = DL.D
    Lambda_train = DL.Lambda
    end = time.time()
    duration = end-start
    if verbose:
        print("Dictionary learning duration =", duration)
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(niter), DL.errors)
        plt.title("Dictionary learning on training set - MSE vs. Iterations.")

    # compute codes for test set
    inv_D = np.linalg.pinv((D.T).dot(D))
    Lambda_test = inv_D.dot((D.T).dot(X_test))
    Lambda_test = np.apply_along_axis(euclidean_proj_simplex, 0, Lambda_test)
    
    # perform prediction and compute accuracy
    Lambda_train = Lambda_train/np.linalg.norm(Lambda_train, axis=0)
    Lambda_test = Lambda_test/np.linalg.norm(Lambda_test, axis=0)
    dot_product = Lambda_test.T.dot(Lambda_train)
    y_pred = y_train[np.argmax(dot_product, axis=1)]
    accuracy = np.sum(y_pred==y_test)/len(y_test)
    if verbose:
        print("Accuracy on test set:", accuracy)
    
    return {'training_idxes': training_idxes, 'testing_idxes': testing_idxes,
            'DL_duration': duration, 'DL_errors': DL.errors,
            'D': D, 'Lambda_train': Lambda_train, 'Lambda_test': Lambda_test,
            'y_pred': y_pred, 'accuracy': accuracy}


# ## Experiments

# In[9]:


X = orl_transport_plan
k = 2

np.random.seed(713)

experiment_transport_plan_k2 = orl_experiment(X, k)


# In[10]:


X = orl_transport_plan
k = 5

np.random.seed(1377)

experiment_transport_plan_k5 = orl_experiment(X, k)


# In[11]:


X = orl_transport_plan
k = 8

np.random.seed(663)

experiment_transport_plan_k8 = orl_experiment(X, k)


# In[12]:


X = orl_transport_plan
k = 10

np.random.seed(1831)

experiment_transport_plan_k10 = orl_experiment(X, k)


# In[13]:


X = orl_transport_plan
k = 20

np.random.seed(514)

experiment_transport_plan_k20 = orl_experiment(X, k)


# In[14]:


X = orl_transport_plan
k = 30

np.random.seed(1737)

experiment_transport_plan_k30 = orl_experiment(X, k)


# In[15]:


X = orl_transport_plan
k = 40

np.random.seed(478)

experiment_transport_plan_k40 = orl_experiment(X, k)


# In[16]:


X = orl_transport_plan
k = 50

np.random.seed(1232)

experiment_transport_plan_k50 = orl_experiment(X, k)


# ## Experiments on natural images

# In[17]:


X = orl_natural_images
k = 2

np.random.seed(1792)

experiment_natural_images_k2 = orl_experiment(X, k)


# In[18]:


X = orl_natural_images
k = 5

np.random.seed(567)

experiment_natural_images_k5 = orl_experiment(X, k)


# In[19]:


X = orl_natural_images
k = 8

np.random.seed(1920)

experiment_natural_images_k8 = orl_experiment(X, k)


# In[20]:


X = orl_natural_images
k = 10

np.random.seed(1767)

experiment_natural_images_k10 = orl_experiment(X, k)


# In[21]:


X = orl_natural_images
k = 20

np.random.seed(48)

experiment_natural_images_k20 = orl_experiment(X, k)


# In[22]:


X = orl_natural_images
k = 30

np.random.seed(21)

experiment_natural_images_k30 = orl_experiment(X, k)


# In[23]:


X = orl_natural_images
k = 40

np.random.seed(1230)

experiment_natural_images_k40 = orl_experiment(X, k)


# In[24]:


X = orl_natural_images
k = 50

np.random.seed(599)

experiment_natural_images_k50 = orl_experiment(X, k)


# In[ ]:




