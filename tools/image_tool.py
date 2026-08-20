import os

import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.visualization import ZScaleInterval, LinearStretch
import skimage.exposure as skie

kernel = Gaussian2DKernel(x_stddev=2)


def CatPSimgMinMax(img_dat):
    medflux = np.median(img_dat)
    madflux = np.median(np.abs(img_dat - medflux))

    lomad, himad = (-2.0, 10.0)  # number of mads to the min and max

    minflux = medflux + lomad * madflux
    maxflux = medflux + himad * madflux

    # midflux = medflux  + 0.5 * (himad + lomad) * madflux
    # radflux = 0.5 * (himad - lomad) * madflux

    return minflux, maxflux


def optimize_image(image_data):
    limg = np.arcsinh(image_data)
    limg = limg / limg.max()
    low, high = np.percentile(limg, (0.25, 99.5))
    opt_img = skie.exposure.rescale_intensity(limg, in_range=(low, high))
    return opt_img


def imshow(img_tensor):
    npimg = img_tensor.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg, origin='lower', cmap=cmaps.get_cmap("Greys"))
    plt.show()
