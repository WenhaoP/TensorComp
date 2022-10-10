from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from colour_demosaicing import masks_CFA_Bayer
from timeit import default_timer as timer
from nonten import nonten_initial_alter

from utils import plot_channel, the_to_q, q_to_the

# @profile
def main():
    shape = (300, 200)
    start = (50, 50)
    image = imread('image/oski.png', 'png')   
    image = image[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1], :]

    bayer_filter = np.stack(masks_CFA_Bayer(shape), axis=-1).astype(float)
    mosaic_image = image * bayer_filter
    mosaic_image[..., 1] *= GREEN_SCAL_COEF

    X = np.argwhere(bayer_filter.flatten() == 1).flatten()
    Y = mosaic_image.flatten()[X]
    r = mosaic_image.shape

    if INITIALIZE:
        if HEURISTIC == 3:
            Vts_init = np.load(f'result/region_shape_{SIZE}_the_3.npy').T
            lamb_init = np.load(f'result/region_shape_{SIZE}_mu_3.npy')
            lamb_sum = lamb_init.sum()
            lamb_init = (lamb_init / lamb_sum)[:, None]
            psi_q_init = np.load(f'result/region_shape_{SIZE}_Psi_3.npy').flatten()[X] / lamb_sum
        elif HEURISTIC == 4:
            Vts_init = np.load(f'result/region_shape_{SIZE}_the_4.npy').T
            lamb_init = np.load(f'result/region_shape_{SIZE}_mu_4.npy')
            lamb_init = (lamb_init / lamb_init.sum())[:, None]
            psi_q_init = np.load(f'result/region_shape_{SIZE}_Psi_4.npy').flatten()[X]
    else:
        Vts_init = np.ones(sum(r)).reshape(-1, 1)
        lamb_init = np.array([1])
        psi_q_init = np.ones_like(X)

    out = nonten_initial_alter(X, Y, r, None, Vts_init, psi_q_init, lamb_init, lpar = LPAR, tol = 1e-6, stop_iter=STOP_ITER, max_altmin_cnt = MAX_ALTMIN_CNT, verbose = True).reshape(image.shape)
    out[..., 1] /= GREEN_SCAL_COEF

    if INITIALIZE:
        np.save(f'result/GSC_{round(GREEN_SCAL_COEF,2)}_INIT_{INITIALIZE}_HEU_{HEURISTIC}_SZ_{SIZE}_LPAR_{LPAR}_SI_{STOP_ITER}_MAC_{MAX_ALTMIN_CNT}', out)
    else:
        np.save(f'result/GSC_{round(GREEN_SCAL_COEF,2)}_INIT_{INITIALIZE}_LPAR_{LPAR}_SI_{STOP_ITER}_MAC_{MAX_ALTMIN_CNT}', out)

if __name__ == '__main__':
    
    INITIALIZE = False
    HEURISTIC = 3
    SIZE = [3, 2]
    GREEN_SCAL_COEF = 1
    LPAR = 1
    STOP_ITER = 100000
    MAX_ALTMIN_CNT = 100

    main()
    