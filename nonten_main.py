import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from colour_demosaicing import masks_CFA_Bayer
from timeit import default_timer as timer
from nonten import nonten_initial, nonten_initial_alter

from utils import plot_channel, the_to_q, q_to_the

@profile
def main():
    shape = (300, 200)
    start = (50, 50)
    image = imread('image/oski.png', 'png')   
    image = image[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1], :]

    bayer_filter = np.stack(masks_CFA_Bayer(shape), axis=-1).astype(float)
    mosaic_image = image * bayer_filter

    X = np.argwhere(bayer_filter.flatten() == 1).flatten()
    Y = mosaic_image.flatten()[X]
    r = mosaic_image.shape

    # Pts_init = np.load('result/Pts_init.npy').astype(np.float32)
    Vts_init = np.load('result/Vts_init.npy').astype(np.float32)
    lamb_init = np.load('result/lamb_init.npy').astype(np.float32)
    psi_q_init = np.load('result/psi_q_init.npy').astype(np.float32)

    out = nonten_initial_alter(X, Y, r, None, Vts_init, psi_q_init, lamb_init, lpar = LPAR, tol = 1e-6, stop_iter=STOP_ITER, max_altmin_cnt = MAX_ALTMIN_CNT, verbose = True)

    np.save(f'result/lpar_{LPAR}_stop_iter_{STOP_ITER}_max_altmin_cnt_{MAX_ALTMIN_CNT}', out)

if __name__ == '__main__':
    
    LPAR = 20000
    STOP_ITER = 10000
    MAX_ALTMIN_CNT = 100

    main()
    