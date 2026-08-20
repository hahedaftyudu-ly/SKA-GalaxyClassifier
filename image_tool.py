import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.visualization import ZScaleInterval, LinearStretch

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


image_root = 'data/too_many_invalid_values/STAR/10.838543p13.991331'
image_fs = os.listdir(image_root)
image_fs_abs_path = [os.path.join(image_root, f) for f in image_fs]
image_fs_abs_path.sort()  # g, i, r, y, z
fig = plt.figure()

fits_f = image_fs_abs_path[0]
print(fits_f)
image_dat = fits.getdata(fits_f)  # max = 779232.3, min = -711.6152

center_region = image_dat[100:140, 100:140]  # max = 2316.12

# vmin, vmax = CatPSimgMinMax(image_dat)  # -117.55644416809082 672.1435680389404
# vmin, vmax = CatPSimgMinMax(center_region)  # -124.03432846069336 728.589786529541
#
# print(vmin, vmax)
# image_dat = np.where(image_dat > vmax, vmax, image_dat)
# # if vmax > image_cube_vmax:
# #     image_cube_vmax = vmax

vmin = np.min(image_dat)
vmax = np.max(image_dat)
# image_dat = (image_dat - vmin) / (vmax - vmin)
image_dat = LinearStretch().__call__(ZScaleInterval().__call__(image_dat))
print(np.max(image_dat), np.min(image_dat))
plt.imshow(image_dat, origin="lower", cmap="Greys")
plt.show()

fig = plt.figure()
plt.hist(image_dat)
plt.show()
#
