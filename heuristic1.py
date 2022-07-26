import numpy as np
import matplotlib.pyplot as plt
import time

### Helper functions for heuristic 1 ###

# alternating maximization oracle
def altmax(r, p, gamma, tol, cmax, c, the_q, Un):
    the = the_q.copy()
    cum_r = np.insert(np.cumsum(r), 0, 0)
    un = Un.shape[0]

    last_cmax = cmax - 1e6
    cnt = 0
    while (True):
        cnt += 1
        the[cum_r[2]:cum_r[3]] = 0
        the[cum_r[2]+np.random.randint(0, 3)] = 1
        for ind in range(p-1):
            pro = c.copy()
            for k in range(p):
                if (ind != k):
                    mask = (the[cum_r[k] + Un[:,k]] == 0)
                    pro[mask] = 0 

            fpro = np.zeros(r[ind])
            for i in range(r[ind]):
                if ind == 0:
                    idx = np.arange(i*600, (i+1)*600)
                    fpro[i] += pro[idx].sum()
                elif ind == 1:
                    idx = np.repeat(np.arange(0, 300) * 600, 3) + np.tile(np.arange(i*3, (i+1)*3), 300)
                    fpro[i] += pro[idx].sum()
                else:
                    idx = np.arange(60000) * 3 + i
                    fpro[i] += pro[idx].sum()

            mask = (fpro >= np.sort(fpro)[r[ind] - gamma[ind]])
            the[cum_r[ind]:cum_r[ind+1]] = (mask).astype(int)
            curr_cmax = np.sum(fpro[mask])

        if (curr_cmax < last_cmax + tol):
            break
        else:
            last_cmax = curr_cmax

    psi = np.ones(un)
    for k in range(p):
        psi = np.multiply(psi, the[cum_r[k] + Un[:,k]])

    return(psi, the, curr_cmax)

### Main bodies of the huristic 1 ###

def heuristic_1(p_0, gamma, lpar=1, tol=1e-4, verbose=False):
    """
    Deflate/decompose the tensor p_0 of the mosaic image
    ---
    Arguments:
    p_0 (np.array): the mosaic image
    gamma (list[scalar]): the l1 norm upper constraint on every theta
    lpar (scalar): the lambda parameter
    tol (scalar): the tolarance used in the alternating maximization

    Returns:
    psi_t (np.array): the flatten version
    Psi_t (np.array): the original version
    mu (list[scalar]): a list of mu_t
    the (list[np.array]): a list of the_t
    """

    # initializing parameters
    r = p_0.shape
    p = len(r)
    cmax = -float('inf')
    Un = np.arange(np.prod(r)) 
    Un = np.unravel_index(Un, r)
    Un = np.vstack(Un).T
    p_t = p_0.flatten()
    un = len(Un)
    N_t = 0
    psi_t = np.zeros(un)
    mu = []
    the = []

    # the deflation algorithm
    while True:
        if verbose:
            print(f'N_t is {N_t}')
        the = np.round(np.random.uniform(0,1,np.sum(r)))
        q_t, the_t, _ = altmax(r, p, gamma, tol, cmax, p_t, the, Un)
        mu_t = np.min((p_t / q_t)[(p_t > 0) & (q_t > 0)])

        if (N_t + mu_t > lpar):
            print(f'Heuristic 1: N_t + mu_t {N_t + mu_t} is larger than lpar {lpar}')
            break      
        
        N_t = N_t + mu_t
        psi_t = psi_t + mu_t * q_t
        p_t = p_t - mu_t * q_t
        the.append(the_t)
        mu.append(mu_t)

        if p_t.sum() == 0:
            break

    # recover the original shape
    Psi_t = psi_t.reshape(r)

    return psi_t, Psi_t, mu, the
