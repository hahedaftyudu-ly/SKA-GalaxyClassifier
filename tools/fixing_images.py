import logging
import os
import shutil

import numpy as np
from astropy.io import fits

from utils import have_source_in_center, remove_nan

galaxy_source_root = os.path.join(os.getcwd(), "data/sources/GALAXY")
star_source_root = os.path.join(os.getcwd(), "data/sources/STAR")
quasar_source_root = os.path.join(os.getcwd(), "data/sources/QSO")

too_many_nan_galaxy = os.path.join(os.getcwd(), 'data/too_many_invalid_values/GALAXY')
too_many_nan_star = os.path.join(os.getcwd(), 'data/too_many_invalid_values/STAR')
too_many_nan_QSO = os.path.join(os.getcwd(), 'data/too_many_invalid_values/QSO')

no_source_in_center_galaxy = os.path.join(os.getcwd(), "data/no_source_in_center/GALAXY")
no_source_in_center_star = os.path.join(os.getcwd(), "data/no_source_in_center/STAR")
no_source_in_center_QSO = os.path.join(os.getcwd(), "data/no_source_in_center/QSO")


def removeNAN(img_types_path=galaxy_source_root):
    if img_types_path.endswith("GALAXY"):
        dest = too_many_nan_galaxy
    elif img_types_path.endswith("QSO"):
        dest = too_many_nan_QSO
    elif img_types_path.endswith("STAR"):
        dest = too_many_nan_star
    else:
        raise ValueError

    for root, dirs, fnames in sorted(os.walk(img_types_path, followlinks=True)):  # here
        for dir in sorted(dirs):
            src_path = os.path.join(root, dir)
            fits_img = [os.path.join(src_path, f) for f in os.listdir(src_path)]

            for f in fits_img:
                img_dat = fits.getdata(f)
                x, y = np.where(np.isnan(img_dat))
                if len(x) > 100:  # drop this sources
                    shutil.move(src_path, dest)  # here
                    print(f"{f} contains {len(x)} nan pixels, discard this source from dataset")
                    break


removeNAN(img_types_path=quasar_source_root)


def check_have_source_in_center(label_path=quasar_source_root):
    if label_path.endswith("GALAXY"):
        dest = no_source_in_center_galaxy
    elif label_path.endswith("QSO"):
        dest = no_source_in_center_QSO
    elif label_path.endswith("STAR"):
        dest = no_source_in_center_star
    else:
        raise ValueError

    for root, dirs, fnames in sorted(os.walk(label_path, followlinks=True)):
        for dir in sorted(dirs):
            src_path = os.path.join(root, dir)
            fits_img = [os.path.join(src_path, f) for f in os.listdir(src_path)]
            image_data_list = [fits.getdata(f) for f in fits_img]
            fixed_image_list = []
            for i in range(len(image_data_list)):
                tmp = remove_nan(image_dat=image_data_list[i])
                fixed_image_list.append(tmp)

            flags = []
            for j in range(len(fixed_image_list)):
                flags.append(have_source_in_center(fixed_image_list[j]))

            flags = np.asarray(flags)
            if np.any(flags):
                print(f"{dir} is a valid source!")
            else:
                logging.warning(f"{dir}: No source has been detected in the image center, discard this "
                                f"source from dataset")
                shutil.move(src_path, dest)

# check_have_source_in_center(label_path="/mnt/DataDisk/Duncan/images44/STAR")
