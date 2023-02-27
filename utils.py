# Utilities for Linearized Wasserstein Dictionary Learning
# September 14, 2022



# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import re

from pysdot.domain_types import ConvexPolyhedraAssembly, ScaledImage
from pysdot import PowerDiagram

# quantiles - histograms conversion
def quantiles_to_histograms(quantiles, b, quantile_range=(-12, 12)):
    n = quantiles.shape[1]
    histograms = np.zeros((b, n))
    for i in range(n):
        histograms[:, i] = np.histogram(quantiles[:, i], b, range=quantile_range)[0]/(2*b)
    return histograms

def histograms_to_quantiles(histograms, b_quantiles, m=2000, quantile_range=(-12, 12)):
    (b, n) = histograms.shape
    quantiles = np.zeros((b_quantiles, n))
    interval = np.linspace(quantile_range[0], quantile_range[1], b)
    for i in range(n):
        unormalized_histogram = m*histograms[:, i]
        sample = []
        for k in range(b):
            sample += int(unormalized_histogram[k])*[interval[k]]
        for k in range(b_quantiles):
            quantiles[k, i] = np.quantile(sample, k/b_quantiles)
    return quantiles

# image processing
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



def images_to_point_clouds(Images, jittering_scale=0):
    '''Converts an array of images into weighted point clouds.
    Inputs:
        Images (np.array): array of grayscale images of format (n, x_length, y_length)
        jittering_scale (float): level of jittering of the pixel coordinate to ensure
                                 stable optimal transport computations
    Returns:
        point_clouds (dict): dictionary of point clouds coordinates
                             (redundant dictionary, adapted to sd-ot)
        masses (dict): dictionary of normalized pixel values (adapted to sd-ot))
    '''
    point_clouds, masses = {}, {}
    (n, x_length, y_length) = Images.shape
     
    # masses of the point cloud (pixel intensities)
    for i in range(n):
        # coordinates of the point cloud (we add a 'jittering' of epsilon to avoid pysdot issues)
        idxes = np.where(Images[i]>0)
        cloud = np.zeros((len(idxes[0]), 2))
        cloud[:, 0] = idxes[1]/(y_length-1)
        cloud[:, 1] = 1 - idxes[0]/(x_length-1)
        if jittering_scale>0:
            epsilon = np.random.normal(scale=jittering_scale, size=cloud.shape)
            cloud += epsilon
        point_clouds[i] = cloud
        nu = Images[i][idxes]
        masses[i] = nu / np.sum(nu)
    
    return point_clouds, masses

# dictionary learning utilities
def plot_atoms(D, width, b, title='', x_range=(-12, 12)):
    colors = ['b', 'g', 'r']
    x_axis = np.linspace(x_range[0], x_range[1], b)
    for i in range(len(D)):
        plt.bar(x_axis, D[i, :], color=colors[i%len(colors)], label='Atom {}'.format(i+1), alpha=0.3, width=width)
    plt.title(title)
    plt.legend()

def observe_reconstructions(original, rec, method, width, nb=5, x_range=(-12, 12)):
    (b, n) = original.shape
    x_axis = np.linspace(x_range[0], x_range[1], b)
    middle = nb//2
    for i in range(nb):
        idx = np.random.randint(n)
        plt.subplot(1, nb, i+1)
        plt.bar(x_axis, original[:, idx], color='g', label='Original', alpha=0.3, width=width)
        plt.bar(x_axis, rec[:, idx], color='r', label='Reconstruction', alpha=0.3, width=width)
        if (i==middle): plt.title('Method = {}'.format(method), fontdict={'fontsize': 20})
        plt.legend()


# simplex projection
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return 
    
# optimal transport
def make_square(box=[0, 0, 1, 1]):
    '''
    Constructs a square domain with uniform measure
    To be passed to the laguerre_* functions
    Args:
        box (list): coordinates of the bottom-left and top-right corners
    Returns:
        domain (pysdot.domain_types): domain
    '''
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain

def laguerre_areas(domain, Y, psi, der=False):
    '''
    Computes the areas of the Laguerre cells intersected with the domain
    Args:
        domain (pysdot.domain_types): domain of the (continuous)
                                      source measure
        Y (np.array): points of the (discrete) target measure
        psi (np.array or list): Kantorovich potentials
        der (bool): wether or not return the Jacobian of the areas
                    w.r.t. psi
    Returns:
        pd.integrals() (list): list of areas of Laguerre cells
    '''
    pd = PowerDiagram(Y, -psi, domain)
    if der:
        N = len(psi)
        mvs = pd.der_integrals_wrt_weights()
        return mvs.v_values, csr_matrix((-mvs.m_values, mvs.m_columns, mvs.m_offsets), shape=(N, N))
    else:
        return pd.integrals()