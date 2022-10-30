import numpy as np
import scipy.sparse as sp
from itertools import product

@profile
def the_to_psi_sparse(the, X, r, cum_r):
    """
    Convert the (concatenated n theta's for constructing a tesnor of order of n) to the psi (projected flatten tensor) in a scipy sparse matrix
    ---
    Arguments:
    X (np.array): the indices of known entries in projected tensor (psi)
    """
    the = the.flatten()

    if the.sum() == 0: # when psi is a zero tensor
        psi_sparse = sp.csc_matrix((len(X), 1))
    elif the.sum() == np.sum(r): # when psi is a one tensor
        psi_one_idx = np.arange(len(X))
        psi_sparse = sp.csc_matrix((np.ones(len(psi_one_idx)), (np.arange(len(X)), np.zeros(len(X)))), shape=(len(X), 1))
    else:
        p = len(r)
        psi_one_idx = np.argwhere(the[cum_r[0]:cum_r[1]])
        for i in range(1, p):
            one_idx = np.argwhere(the[cum_r[i]:cum_r[i+1]])
            psi_one_idx = np.hstack((np.tile(psi_one_idx, (len(one_idx), 1)), np.repeat(one_idx, len(psi_one_idx))[:,None]))

        psi_one_idx = np.ravel_multi_index(psi_one_idx.T, r)

        psi_sparse = sp.csc_matrix((np.ones(len(psi_one_idx)), (psi_one_idx, np.zeros_like(psi_one_idx))), shape=(np.prod(r), 1))
        psi_sparse = psi_sparse[X]

        c = np.argwhere(np.isin(X, psi_one_idx)).flatten()
        psi_sparse_1 = sp.csc_matrix((np.ones(len(c)), (c, np.zeros_like(c))), shape=(len(X), 1))

    return psi_sparse