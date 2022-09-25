import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy.sparse as sp
from itertools import product

def plot_channel(Psi, image):
    Psi_zeros = np.zeros(image.shape)
    Psi_r = Psi_zeros.copy()
    Psi_g = Psi_zeros.copy()
    Psi_b = Psi_zeros.copy()
    Psi_r[..., 0] = Psi[..., 0]
    Psi_g[..., 1] = Psi[..., 1]
    Psi_b[..., 2] = Psi[..., 2]

    image_zeros = np.zeros(image.shape)
    image_r = image_zeros.copy()
    image_g = image_zeros.copy()
    image_b = image_zeros.copy()
    image_r[..., 0] = image[..., 0]
    image_g[..., 1] = image[..., 1]
    image_b[..., 2] = image[..., 2]

    fig, axes = plt.subplots(1, 6, figsize=(25, 25), constrained_layout=True)
    axes[0].imshow(Psi_r)
    axes[1].imshow(image_r)
    axes[2].imshow(Psi_g)
    axes[3].imshow(image_g)
    axes[4].imshow(Psi_b)
    axes[5].imshow(image_b)
    plt.show()

def q_to_the(q):
    """
    Convert a vertex of a 0-1 polytope (which has a nonnegative rank of 1) from the tensor form to the simple form.
    i.e. q = the[0:r1] x the[r1:r1+r2] x ... x the[-rp:]
    ---
    Arguments:
    q (np.array): the vertex in the tensor form

    Returns:
    the: the vertex in the tensor form
    """
    r = q.shape
    cum_r = np.insert(np.cumsum(r), 0, 0)
    the = np.zeros(np.sum(r))

    one_idx = np.argwhere(q == 1)

    for i in range(len(r)):
        pos = np.unique(one_idx[:, i])
        the[pos + cum_r[i]] = 1
    
    the = the.astype(int)

    return the

def the_to_q(the, r):
    """
    Convert a vertex of a 0-1 polytope (which has a nonnegative rank of 1) from  the simple form to the tensor form.
    i.e. q = the[cum_r[0]:cum_r[1]] x the[cum_r[1]:cum_r[2]] x ... x the[cum_r[len(r)]:cum_r[len(r)+1]]
    ---
    Arguments:
    theta: the vertex in the tensor form

    Returns:
    q (np.array): the vertex in the tensor form
    """
    cum_r = np.insert(np.cumsum(r), 0, 0)
    q = the[cum_r[0]:cum_r[1]]

    for i in range(1, len(r)):
        q = np.tensordot(q, the[cum_r[i]:cum_r[i+1]], axes=0)

    return q

def Vts_to_Pts(Vts, X, r, cum_r):
    """
    Convert Vts to Pts which is a scipy sparse matrix
    """
    def find_idx(c, c_idx):
        """
        Find the row index of nonzero entries in each column of Pts_sparse 
        """
        vts_one_idx = np.argwhere(c == 1)
        idx = list(
            product(
                (vts_one_idx[vts_one_idx < cum_r[1]]),
                (vts_one_idx[(vts_one_idx >= cum_r[1]) & (vts_one_idx < cum_r[2])] - cum_r[1]),
                (vts_one_idx[(vts_one_idx >= cum_r[2]) & (vts_one_idx < cum_r[3])] - cum_r[2])
            )
        )
        flat_idx = np.ravel_multi_index(np.array(idx).T, r)
        out = np.argwhere(np.isin(X, flat_idx))
        out = np.hstack([out, np.ones(len(out), dtype=int).reshape(-1, 1) * c_idx])
        return out
    
    if Vts.sum() == 0:
        Pts_sparse = sp.csc_matrix((len(X), Vts.shape[1]))
    else:
        ones_idx = np.vstack(list(map(find_idx, Vts.T, np.arange(Vts.shape[1]))))
        Pts_sparse = sp.csc_matrix((np.ones(len(ones_idx)), (ones_idx[:, 0], ones_idx[:, 1])), shape=(len(X), Vts.shape[1]))

    return Pts_sparse