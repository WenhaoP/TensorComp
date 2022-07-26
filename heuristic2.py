import numpy as np
import matplotlib.pyplot as plt
import time

from utils import q_to_the

### Helper functions for heuristic 2 ###

def place_block_1(rows, cols, c, p_t, block_shape):
    """
    Put a "block" of 1's in a zero tensor at the same position of the minimal positive entry in the region of p_t defined by rows and cols
    ---
    Arguments:
    rows (sequence[int]): row indices of the left and right boundaries of the region
    cols (sequence[int]): column indices of the left and right boundaries of the region
    c (int): the channel index
    p_t (int): the tensor
    block_shape: the shape of the block

    Returns:
    q_t: the zero tensor that contains a block of 1's
    mu_t (scalar): the value of the minimal positive entry
    mu_t_idx (tuple(int)): the index of the minimal positive entry
    """
    if not isinstance(block_shape, np.ndarray):
        block_shape = np.array(block_shape)

    # find the region    
    region = p_t[..., c].copy() # c-th channel
    region[region <= 0] = np.inf # for finding the minimal positive entry easily
    region = region[rows[0]:rows[1], cols[0]:cols[1]]
    
    # find the value and index of the minimal positive entry 
    mu_t_idx = np.unravel_index(np.argmin(region), region.shape)
    mu_t = region[mu_t_idx]
    if np.isinf(mu_t):
        mu_t = 0
    mu_t_idx = (mu_t_idx[0] + rows[0], mu_t_idx[1] + cols[0], c)

    # place the "block" in the zero tensor
    left = block_shape // 2
    right = block_shape - left
    q_t = np.zeros(p_t.shape)
    q_t[mu_t_idx[0] - left[0]: mu_t_idx[0] + right[0], mu_t_idx[1] - left[1]: mu_t_idx[1] + right[1], c] = 1

    # q_t = np.zeros(p_t.shape)
    # q_t[rows[0]:rows[1], cols[0]:cols[1], c] = mu_t

    return q_t, mu_t, mu_t_idx

def recover_channel_1(p_0, block_shape, c, min_n_pix, lpar, verbose=False):
    """
    Reconstruct the c-th channel of the mosaic image p_9
    ---
    Arguments:
    p_0 (np.array): the mosaic image
    block_shape (sequence[scalar]): the shape of the block
    c (int): the channel index
    min_n_pix (int): the minimal number of pixels in a considered region 
    lpars (sequence[scalar]): the lambda parameters for each channel

    Returns:
    Psi_t (np.array): the recovered channel
    mu (list[scalar]): a list of mu_t
    the (list[np.array]): a list of the_t
    """
    r = p_0.shape
    P_t = p_0
    N_t = 0
    Psi_t = np.zeros(r)
    mu = []
    the = []

    row_boundary = [0, r[0]]
    col_boundary = [0, r[1]]

    while True:

        if verbose:
            print(f'N_t is {N_t}')
            Psi_zeros = np.zeros(r)
            Psi_r = Psi_zeros.copy()
            Psi_g = Psi_zeros.copy()
            Psi_b = Psi_zeros.copy()
            Psi_r[..., 0] = Psi_t[..., 0]
            Psi_g[..., 1] = Psi_t[..., 1]
            Psi_b[..., 2] = Psi_t[..., 2]
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            axes[0].imshow(Psi_r)
            axes[1].imshow(Psi_g)
            axes[2].imshow(Psi_b)
            plt.show()
            time.sleep(1)
            plt.close(fig)

        num_boundary = len(row_boundary)

        # place a "block" in each region
        for i in range(num_boundary-1):
            for j in range(num_boundary-1):

                # stop when a region is too small
                if (row_boundary[i+1] - row_boundary[i]) * (col_boundary[j+1] - col_boundary[j]) <= min_n_pix:
                    print(f"Heuristic 2: The region in channel {c} is too small: {(row_boundary[i+1] - row_boundary[i]) * (col_boundary[j+1] - col_boundary[j])} pixels.")
                    return Psi_t, mu, the
                
                # place a block in the region 
                q_t, mu_t, mu_t_idx = place_block_1((row_boundary[i], row_boundary[i+1]), (col_boundary[j], col_boundary[j+1]), c, P_t, block_shape)
                the_t = q_to_the(q_t)

                # guarantee the gauge norm constraint
                if N_t + mu_t > lpar:
                    print(f'Heuristic 2: N_t + mu_t {N_t + mu_t} is larger than lpar {lpar} in channel {c}')
                    return Psi_t, mu, the
                else:
                    N_t = N_t + mu_t
                    Psi_t = Psi_t + mu_t * q_t
                    P_t = P_t - mu_t * q_t
                    mu.append(mu_t)
                    the.append(the_t)      
            
        # bisect each region
        for k in range(1, num_boundary):
            row_boundary.append(row_boundary[k-1] + (row_boundary[k] - row_boundary[k-1]) // 2)
            col_boundary.append(col_boundary[k-1] + (col_boundary[k] - col_boundary[k-1]) // 2)
        
        row_boundary.sort()
        col_boundary.sort()

### Main bodies of the huristic 2 ###

def heuristic_2(p_0, block_shape, min_n_pix, lpars, verbose=False):
    """
    Deflate/decompose the tensor p_0 of the mosaic image
    ---
    Arguments:
    p_0 (np.array): the mosaic image
    block_shape (sequence[scalar]): the shape of the block
    min_n_pix (int): the minimal number of pixels in a considered region 
    lpars (sequence[scalar]): the lambda parameters for each channel

    Returns:
    Psi (np.array): the recovered image
    mu (list[scalar]): a list of mu_t
    the (list[np.array]): a list of the_t
    """
    Psi = np.zeros(p_0.shape)
    mu = []
    the = []

    # recover each color channel separately
    for c in range(3):
        Psi_c, mu_c, the_c = recover_channel_1(p_0, block_shape, c, min_n_pix, lpars[c], verbose)
        Psi = Psi + Psi_c
        mu.extend(mu_c)
        the.extend(the_c)
    
    return Psi, mu, the