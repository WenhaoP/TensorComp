import numpy as np
import scipy.sparse as sp
from itertools import product

def Vts_to_Pts_sparse(Vts, X, r, cum_r):
    """
    Convert Vts to Pts which is a scipy sparse matrix
    ---
    Arguments:
    X (np.array): the indices of known entries after projection
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