import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from colour_demosaicing import masks_CFA_Bayer
from timeit import default_timer as timer

from heuristic1 import heuristic_1
from heuristic2 import heuristic_2
from heuristic3 import heuristic_3
from heuristic4 import heuristic_4

if __name__ == '__main__':

    shape = (300, 200)
    start = (50, 50)
    image = imread('image/oski.png', 'png')   
    image = image[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1], :]

    bayer_filter = np.stack(masks_CFA_Bayer(shape), axis=-1).astype(float)
    mosaic_image = image * bayer_filter

    lpar1 = 100
    lpars2 = [3000, 2000, 2000]
    lpars3 = [5000, 4000, 3000]
    lpars4 = [10000/3, 10000/3, 10000/3]

    block_shape = [3, 2]
    region_shape = [3, 2]
    min_n_pix = 4

    # start_1 = timer()
    # _, Psi_1, mu_1, the_1 = heuristic_1(mosaic_image, [12, 8, 1], lpar=lpar1, tol=1e-2, verbose=False)
    # np.save("result/mu_1", mu_1)
    # np.save("result/the_1", the_1)
    # end_1 = timer()

    start_2 = timer()
    Psi_2, mu_2, the_2 = heuristic_2(p_0=mosaic_image, block_shape=block_shape, min_n_pix=min_n_pix, lpars=lpars2, verbose=False)
    np.save("result/mu_2", mu_2)
    np.save("result/the_2", the_2)
    end_2 = timer()

    start_3 = timer()
    Psi_3, mu_3, the_3 = heuristic_3(p_0=mosaic_image, region_shape=region_shape, lpars=lpars3, verbose=False)
    np.save("result/mu_3", mu_3)
    np.save("result/the_3", the_3)
    end_3 = timer()

    start_4 = timer()
    Psi_4, mu_4, the_4 = heuristic_4(p_0=mosaic_image, min_n_pix=min_n_pix, lpars=lpars4, verbose=False)
    np.save("result/mu_4", mu_4)
    np.save("result/the_4", the_4)
    end_4 = timer()

    # print(f'Heuristic 1: Sum of mu is {sum(mu_1)}. Store {len(the_1)} vertices. Run time is {end_1 - start_1}')
    print(f'Heuristic 2: Sum of mu is {sum(mu_2)}. Store {len(the_2)} vertices. Block shape is {block_shape}. Minimal region area is {min_n_pix}. Run time is {end_2 - start_2}')
    print(f'Heuristic 3: Sum of mu is {sum(mu_3)}. Store {len(the_3)} vertices. Region shape is {region_shape}. Run time is {end_3 - start_3}')
    print(f'Heuristic 4: Sum of mu is {sum(mu_4)}. Store {len(the_4)} vertices. Minimal region area is {min_n_pix}. Run time is {end_4 - start_4}')

    fig, axes = plt.subplots(1, 6, figsize=(30, 8))
    axes[0].imshow(image)
    axes[1].imshow(mosaic_image)
    # axes[2].imshow(Psi_1)
    axes[3].imshow(Psi_2)
    axes[4].imshow(Psi_3)
    axes[5].imshow(Psi_4)

    axes[0].set_title('Original image')
    axes[1].set_title('Mosaic image')
    axes[2].set_title('Heuristic 1')
    axes[3].set_title('Heuristic 2')
    axes[4].set_title('Heuristic 3')
    axes[5].set_title('Heuristic 4')
    fig.savefig('comparison.png')