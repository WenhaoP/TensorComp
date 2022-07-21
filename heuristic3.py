import numpy as np
import matplotlib.pyplot as plt
import time

### Helper functions for heuristic 3 ###

def place_block_2(rows, cols, c, p_t, region_shape):
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
    if not isinstance(region_shape, np.ndarray):
        region_shape = np.array(region_shape)

    # find the region    
    region = p_t[..., c].copy() # c-th channel
    region = region[rows[0]:rows[1], cols[0]:cols[1]]

    mu_t = np.mean(region[region != 0])
    q_t = np.zeros(p_t.shape)
    q_t[rows[0]:rows[1], cols[0]:cols[1], c] = 1

    return q_t, mu_t

def recover_channel_2(p_0, region_shape, c, lpar, verbose=False):
    """
    Reconstruct the c-th channel of the mosaic image p_9
    ---
    Arguments:
    p_0 (np.array): the mosaic image
    block_shape (sequence[scalar]): the shape of each region
    c (int): the channel index
    min_n_pix (int): the minimal number of pixels in a considered region 
    lpars (sequence[scalar]): the lambda parameters for each channel

    Returns:
    Psi_t (np.array): the recovered channel
    mu (list[scalar]): a list of mu_t
    q (list[np.array]): a list of q_t
    """
    r = p_0.shape
    P_t = p_0
    N_t = 0
    Psi_t = np.zeros(r)
    mu = []
    q = []

    row_boundary = np.arange(0, r[0]+0.01, region_shape[0]).astype(int)
    col_boundary = np.arange(0, r[1]+0.01, region_shape[1]).astype(int)
    num_boundary = len(row_boundary)

    # place a "block" in each region
    for i in range(num_boundary-1):
        for j in range(num_boundary-1):

            if verbose:
                print(f'N_t is {N_t}')
                Psi_zeros = np.zeros(r)
                Psi_r = Psi_zeros.copy()
                Psi_g = Psi_zeros.copy()
                Psi_b = Psi_zeros.copy()
                Psi_r[..., 0] = Psi_t[..., 0]
                Psi_g[..., 1] = Psi_t[..., 1]
                Psi_b[..., 2] = Psi_t[..., 2]
                fig, axes = plt.subplots(1, 3, figsize=(20, 8))
                axes[0].imshow(Psi_r)
                axes[1].imshow(Psi_g)
                axes[2].imshow(Psi_b)
                plt.show()
                time.sleep(0.5)
                plt.close(fig)

            q_t, mu_t = place_block_2((row_boundary[i], row_boundary[i+1]), (col_boundary[j], col_boundary[j+1]), c, P_t, region_shape)

            # guarantee the gauge norm constraint
            if N_t + mu_t > lpar:
                print(f'Heuristic 3: N_t + mu_t {N_t + mu_t} is larger than lpar {lpar} in channel {c}')
                return Psi_t, mu, q
            else:
                N_t = N_t + mu_t
                Psi_t = Psi_t + mu_t * q_t
                mu.append(mu_t)
                q.append(q_t)      

    return Psi_t, mu, q

### Main bodies of the huristic 3 ###

def heuristic_3(p_0, region_shape, lpars, verbose=False):
    """
    Deflate/decompose the tensor p_0 of the mosaic image
    ---
    Arguments:
    p_0 (np.array): the mosaic image
    block_shape (sequence[scalar]): the shape of each region
    min_n_pix (int): the minimal number of pixels in a considered region 
    lpars (sequence[scalar]): the lambda parameters for each channel

    Returns:
    Psi (np.array): the recovered image
    mu (list[scalar]): a list of mu_t
    q (list[np.array]): a list of q_t
    """
    Psi = np.zeros(p_0.shape)
    mu = []
    q = []

    # recover each color channel separately
    for c in range(3):
        Psi_c, mu_c, q_c = recover_channel_2(p_0, region_shape, c, lpars[c], verbose)
        Psi = Psi + Psi_c
        mu.extend(mu_c)
        q.extend(q_c)
    
    return Psi, mu, q