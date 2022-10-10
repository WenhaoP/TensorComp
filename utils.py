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
        coordinates = []
        for i in range(1, len(r)+1): # find the coordinates of one entries
            coordinates.append((vts_one_idx[vts_one_idx < cum_r[i]] - cum_r[i-1]).tolist())
            vts_one_idx = np.delete(vts_one_idx, np.argwhere(vts_one_idx < cum_r[i]))
        
        # idx = list(
        #     product(
        #         (vts_one_idx[vts_one_idx < cum_r[1]]),
        #         (vts_one_idx[(vts_one_idx >= cum_r[1]) & (vts_one_idx < cum_r[2])] - cum_r[1]),
        #         (vts_one_idx[(vts_one_idx >= cum_r[2]) & (vts_one_idx < cum_r[3])] - cum_r[2])
        #     )
        # )
        idx = list(product(*coordinates))
        flat_idx = np.ravel_multi_index(np.array(idx).T, r)
        out = np.argwhere(np.isin(X, flat_idx))
        out = np.hstack([out, np.ones(len(out), dtype=int).reshape(-1, 1) * c_idx])
        return out
    
    if Vts.sum() == 0: # when Pts is a zero tensor
        Pts_sparse = sp.csc_matrix((len(X), Vts.shape[1]))
    elif Vts.sum() == np.sum(r): # when Pts is a one tensor
        ones_idx = np.array(list(product(range(len(X)), range(Vts.shape[1]))))
        Pts_sparse = sp.csc_matrix((np.ones(len(ones_idx)), (ones_idx[:, 0], ones_idx[:, 1])), shape=(len(X), Vts.shape[1]))
    else:
        ones_idx = np.vstack(list(map(find_idx, Vts.T, np.arange(Vts.shape[1]))))
        Pts_sparse = sp.csc_matrix((np.ones(len(ones_idx)), (ones_idx[:, 0], ones_idx[:, 1])), shape=(len(X), Vts.shape[1]))

    return Pts_sparse