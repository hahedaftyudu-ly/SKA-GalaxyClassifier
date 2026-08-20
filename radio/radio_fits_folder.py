import os

import numpy as np
from astropy.io import fits
from torchvision.datasets import DatasetFolder

from tools.utils import remove_nan
from tools.image_tool import optimize_image


class FitsImageFolder(DatasetFolder):
    EXTENSIONS = ('.fits',)

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None):
        if loader is None:
            loader = self.__fits_loader

        super(FitsImageFolder, self).__init__(root, loader, self.EXTENSIONS,
                                              transform=transform,
                                              target_transform=target_transform)

    @staticmethod
    def __fits_loader(filename):
        image_dat = fits.getdata(filename)
        ################# Step 1. remove nan or np.inf by interplating new value ######################
        image_dat = remove_nan(image_dat)
        ################# Step 2. stretch ######################
        image_dat = optimize_image(image_dat)

        image_dat = image_dat.astype(np.float32)
        return image_dat
