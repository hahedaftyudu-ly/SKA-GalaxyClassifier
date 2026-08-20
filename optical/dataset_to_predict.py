import math
import os
import warnings

import numpy as np
import pandas as pd
from astropy.utils.exceptions import AstropyWarning
from numpy.linalg import norm

warnings.simplefilter('ignore', category=AstropyWarning)
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from skimage.transform import resize

from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
from utils import remove_nan, optimize_image, normalize, get_sigma_clip, cal_luptitude



class PredictingDataset(Dataset):
    def __init__(self, opt_root_dir, ps_positive_samples_csv, ps_negative_samples_csv,
                 img_transforms=None):
        self.opt_root_dir = opt_root_dir
        self.ps_background_cutouts = os.listdir(opt_root_dir)

        self.ps_p_samples = pd.read_csv(ps_positive_samples_csv)
        self.ps_n_samples = pd.read_csv(ps_negative_samples_csv)

        self.df_samples = pd.concat([self.ps_p_samples, self.ps_n_samples], ignore_index=True)
        self.df_samples = self.df_samples.sample(frac=1)

        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.df_samples)


    def create_PS_sample_cutout(self, opt_sample: pd.Series):
        radio_component_name = opt_sample["VLASS_component_name"]
        ps_background_cutout_path = os.path.join(self.opt_root_dir, radio_component_name)
        fits_imgs = os.listdir(ps_background_cutout_path)
        fits_imgs.sort()
        fits_path = [os.path.join(ps_background_cutout_path, img) for img in fits_imgs]
        five_band_img_cutouts = []

        ra, dec = opt_sample['Pan-STARRS_RAJ2000'], opt_sample['Pan-STARRS_DEJ2000']
        for fits_file in fits_path:
            image = fits.getdata(fits_file).squeeze()
            header = fits.getheader(fits_file)
            image[np.isnan(image)] = 0.0
            pos = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
            cutout = Cutout2D(image, position=pos, size=240, wcs=WCS(header).celestial)
            five_band_img_cutouts.append(cutout)

        return five_band_img_cutouts

    def create_imgcube(self, img_data_list):
        img_cube = np.stack(img_data_list, axis=2)
        return img_cube



    def __getitem__(self, idx):
        opt_sample: pd.Series = self.df_samples.iloc[idx]
        radio_component_name = opt_sample["VLASS_component_name"]

        PS_sample_cutouts = self.create_PS_sample_cutout(opt_sample)
        PS_sample_img_list = [cutout.data for cutout in PS_sample_cutouts]
        PS_sample_img_cube = np.stack(PS_sample_img_list, axis=2)

        PS_bg_cutouts = self.getPSBackgroundCutout(opt_sample)

        PS_bg_imgcube = np.stack(PS_bg_cutouts, axis=2)

        w1flux = opt_sample['w1flux']
        w2flux = opt_sample['w2flux']
        m1, m2 = cal_luptitude(w1flux, w2flux)
        wise_magnitude_info = np.asarray([m1, m2])

        ps_sample_pixel_position_info = PS_sample_cutouts[0].position_original
        ps_sample_pixel_position_info = np.asarray(ps_sample_pixel_position_info)


        label = opt_sample["PS_class"]
        if label == 'P':
            label = 1
        else:
            label = 0

        ps_id = opt_sample['Pan-STARRS_objID']
        ps_ra = opt_sample['Pan-STARRS_RAJ2000']
        ps_dec = opt_sample['Pan-STARRS_DEJ2000']

        cname = radio_component_name[len('VLASS1QLCIR J'):]
        if cname.__contains__('-'):
            rra, rdec = cname.split('-')
            VLASS_name = (0, np.float(rra), np.float(rdec))
        else:
            rra, rdec = cname.split('+')
            VLASS_name = (1, np.float(rra), np.float(rdec))

        return PS_bg_imgcube, PS_sample_img_cube, wise_magnitude_info, ps_sample_pixel_position_info, label, (ps_id, ps_ra, ps_dec), VLASS_name
