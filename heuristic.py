import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from colour_demosaicing import masks_CFA_Bayer


# alternating maximization oracle
@profile
def altmax(r, p, gamma, tol, cmax, c, the_q, Un):
    the = the_q.copy()
    cum_r = np.insert(np.cumsum(r), 0, 0)
    un = Un.shape[0]

    last_cmax = cmax - 1e6
    cnt = 0
    while (True):
        cnt += 1
        for ind in range(p):
            pro = c.copy()
            for k in range(p):
                if (ind != k):
                    #pro = np.multiply(pro, the[cum_r[k] + Un[:,k]])
                    mask = (the[cum_r[k] + Un[:,k]] == 0)
                    pro[mask] = 0 

            fpro = np.zeros(r[ind])
            # Original
            for k in range(un):
                fpro[Un[k,ind]] += pro[k]

            # # Speed up 1
            # for k in np.argwhere(pro != 0):
            #     fpro[Un[k,ind]] += pro[k]
            
            # # Speed up 2
            # for i in range(r[ind]):
            #     fpro[i] += pro[np.argwhere(Un[:, ind] == i)].sum()
            
            # # Speed up 3
            # fpro[Un[:, ind]] += pro

            mask = (fpro >= np.sort(fpro)[r[ind] - gamma[ind]])
            #if ind < 2:
            #    mask = (fpro >= np.sort(fpro)[r[ind] - mu[ind]])
            #else:
            #    mask = np.full(r[ind], True)
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

# @profile    
def heuristic(p_0, filter, gamma, proj=False, lpar=1, tol=1e-4):
    """
    Deflate/decompose the tensor p_0 of the mosaic image
    ---
    Arguments:
    p_0 (np.array): the mosaic image
    filter (np.array): the filter used to mosaic the image
    gamma (list[scalar]): the l1 norm upper constraint on every theta
    proj (bool): working in proj_U(C_lpar) if True
    lpar (scalar): the lambda parameter
    tol (scalar): the tolarance used in the alternating maximization

    Returns:
    psi_t (np.array): the flatten version
    Psi_t (np.array): the original version
    mu (list[scalar]): a list of mu_t
    q (list[np.array]): a list of q_t
    """

    # initializing parameters
    r = p_0.shape
    p = len(r)
    cmax = -float('inf')
    if proj:
        Un = np.argwhere(filter == 1)      
        p_t = p_0.flatten()[bayer_filter.flatten() == 1]
    else:
        Un = np.arange(np.prod(r)) 
        Un = np.unravel_index(Un, r)
        Un = np.vstack(Un).T
        p_t = mosaic_image.flatten()
    un = len(Un)
    N_t = 0
    psi_t = np.zeros(un)
    mu = []
    q = []

    # the deflation algorithm
    while (N_t <= lpar):
        print(N_t)
        the = np.round(np.random.uniform(0,1,np.sum(r)))
        q_t, _, _ = altmax(r, p, gamma, 0.01, cmax, p_t, the, Un)
        q.append(q_t)
        mu_t = np.min((p_t / q_t)[(p_t > 0) & (q_t > 0)])
        mu.append(mu_t)
        p_t = p_t - mu_t * q_t
        psi_t = psi_t + mu_t * q_t
        N_t = N_t + mu_t

        if p_t.sum() == 0: # the tensor has been fully decomposed
            break
    
    # recover the original shape
    if proj:
        Psi_t = np.zeros(image.shape)
        np.put(Psi_t, np.ravel_multi_index(Un.T.tolist(), image.shape), psi_t)
    else:
        Psi_t = psi_t.reshape(r)

    return psi_t, Psi_t, mu, q


if __name__ == '__main__':
    shape = (300, 200)
    start = (50, 50)
    image = imread('image/oski.png', 'png')   
    image = image[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1], :]

    bayer_filter = np.stack(masks_CFA_Bayer(shape), axis=-1).astype(float)
    mosaic_image = image * bayer_filter

    psi_t_unproj, Psi_t_unproj, mu_unproj, q_unproj = heuristic(mosaic_image, bayer_filter, [3, 2, 1], proj=False, lpar=1)

    fig, ax = plt.subplots()
    ax.imshow(Psi_t_unproj)
    fig.savefig('recover.png')