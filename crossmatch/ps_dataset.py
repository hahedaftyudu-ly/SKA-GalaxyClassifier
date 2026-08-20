import math
import os
import warnings

import numpy as np
import pandas as pd
from astropy.utils.exceptions import AstropyWarning
from scipy import ndimage as ndi
from numpy.linalg import norm
from skimage import filters, morphology, graph, measure
from skimage.filters import thresholding

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




class FitsImageSet(Dataset):
    def __init__(self, radio_root_dir, opt_root_dir, ps_positive_samples_csv, ps_negative_samples_csv,
                 img_transforms=None):
        self.opt_root_dir = opt_root_dir
        self.radio_root_dir = radio_root_dir
        self.ps_background_cutouts = os.listdir(opt_root_dir)
        self.radio_sources = os.listdir(radio_root_dir)
        # Allow mismatched counts (some components may only have radio or only optical)
        if len(self.ps_background_cutouts) != len(self.radio_sources):
            import warnings
            warnings.warn(f"Radio ({len(self.radio_sources)}) and PS ({len(self.ps_background_cutouts)}) counts differ")

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
            image = remove_nan(image)
            pos = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
            cutout = Cutout2D(image, position=pos, size=240, wcs=WCS(header).celestial)
            five_band_img_cutouts.append(cutout)

        return five_band_img_cutouts

    def create_imgcube(self, img_data_list):
        img_cube = np.stack(img_data_list, axis=2)
        return img_cube

    def preprocess_radio_image(self, radio_component):
        radio_img_path = os.path.join(self.radio_root_dir, f"{radio_component}.fits")
        raw_image = fits.getdata(radio_img_path).squeeze()
        image = remove_nan(raw_image)
        image = get_sigma_clip(image)
        image = rescale_intensity(image, out_range=(0, 1))

        t0, t1, t2 = thresholding.threshold_multiotsu(image, classes=4)
        mask = (image > t0)
        vessels = filters.sato(image, sigmas=range(1, 10)) * mask
        high = 1.3 * np.median(image)
        low = 0.7 * np.median(image)

        thresholded = thresholding.apply_hysteresis_threshold(vessels, low=0.0, high=0.0)
        labeled = ndi.label(thresholded)[0]
        largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))

        binary = labeled == largest_nonzero_label
        skeleton = morphology.skeletonize(binary)

        centroid_y, centroid_x = measure.centroid(labeled > 0)

        g, nodes = graph.pixel_graph(skeleton, connectivity=2)

        try:
            px, distances = graph.central_pixel(
                g, nodes=nodes, shape=skeleton.shape, partition_size=100
            )
        except ValueError as ve:
            if centroid_x in range(108) and centroid_y in range(108):
                px = [0, 0]
            elif centroid_x in range(108) and centroid_y in range(108, 217):
                px = [216, 0]
            elif centroid_x in range(108, 217) and centroid_y in range(108):
                px = [0, 216]
            else:
                px = [216, 216]


        return image, centroid_x, centroid_y, px[1], px[0]

    def __getitem__(self, idx):
        opt_sample: pd.Series = self.df_samples.iloc[idx]
        radio_component_name = opt_sample["VLASS_component_name"]

        radio_img_dat, centroid_x, centroid_y, geodesic_x, geodesic_y = self.preprocess_radio_image(radio_component_name)

        PS_sample_cutouts = self.create_PS_sample_cutout(opt_sample)
        PS_sample_img_list = [cutout.data for cutout in PS_sample_cutouts]
        PS_sample_img_cube = np.stack(PS_sample_img_list, axis=2)
        PS_sample_img_cube = optimize_image(PS_sample_img_cube)

        w1flux = opt_sample['w1flux']
        w2flux = opt_sample['w2flux']
        m1, m2 = cal_luptitude(w1flux, w2flux)
        wise_magnitude_info = np.asarray([m1, m2])

        ps_sample_pixel_position_info = PS_sample_cutouts[0].position_original
        ps_sample_pixel_position_info = np.asarray(ps_sample_pixel_position_info)

        centroid_pos = (4 * centroid_x, 4 * centroid_y)
        centroid_pos = np.asarray(centroid_pos)

        geodesic_pos = (4 * geodesic_x, 4 * geodesic_y)
        geodesic_pos = np.asarray(geodesic_pos)



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
            VLASS_name = (0, float(rra), float(rdec))
        else:
            rra, rdec = cname.split('+')
            VLASS_name = (1, float(rra), float(rdec))

        return radio_img_dat, PS_sample_img_cube, wise_magnitude_info, ps_sample_pixel_position_info, centroid_pos, geodesic_pos, label, (ps_id, ps_ra, ps_dec), VLASS_name
