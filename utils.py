import logging
import math
import os
import shutil

import astropy.table
import numpy as np
import pandas as pd
import seaborn as sn
import skimage.exposure as skie
import torch
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clip
from photutils.segmentation import detect_threshold, detect_sources
from sklearn.metrics import confusion_matrix

kernel = Gaussian2DKernel(x_stddev=1)

import matplotlib.pyplot as plt


def check_if_is_five(source_dir="data/sources/GALAXY"):
    root_path = os.path.join(os.getcwd(), source_dir)
    sources = [os.path.join(root_path, source) for source in os.listdir(root_path)]
    for s in sources:
        if not os.path.isdir(s):
            continue
        img_arr = os.listdir(s)
        if len(img_arr) != 5:
            print(f"{s} don't have 5 fits files")
            raise ValueError

    print("check finished")


# check_if_is_five(source_dir="data/test_sources/GALAXY")


def build_confusion_matrix(net, y_pred, y_ground_truth, dataloader):
    # iterate over validation data
    for inputs, labels in dataloader:
        output = net(inputs)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        labels = labels.data.cpu().numpy()
        y_ground_truth.extend(labels)

        # constant for classes
        classes = ('GALAXY', 'QSO', 'STAR')

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_ground_truth, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('output.png')


def rmnan(filename):
    image = fits.getdata(filename)
    header = fits.getheader(filename)
    for i in range(5):
        single_channel = image[i, :, :]
        x, y = np.where(np.isnan(single_channel))
        logging.warning(f"nan pixels: {x}")
        if len(x) > 50:  # drop this sources
            os.remove(filename)
            logging.warning(f"{filename} has been removed")
            return None
        elif len(x) == 0:
            continue
        else:  # filter this image
            fixed_img = interpolate_replace_nans(single_channel, kernel)
            image[i, :, :] = fixed_img
    fits.update(filename, image, header=header)
    return image


def remove_nan(image_dat):  # shape: [5, 240, 240]
    row, col = np.where(np.isnan(image_dat))
    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)
    while len(row) > 0:
        # replace bad data with values interpolated from their neighbors
        image_dat = interpolate_replace_nans(image_dat, kernel)
        row, col = np.where(np.isnan(image_dat))
    return image_dat


def classify_downloaded_sources(dir):
    T = astropy.table.Table.read("/home/duncan/PycharmProjects/1XSDSS_DR16/data/DuncanSDSSdata.tbl",
                                 format='ipac')
    ralist = list(T['ra'])
    declist = list(T['dec'])
    classes = list(T['class'])

    positions = []
    for i in range(len(ralist)):
        positions.append((ralist[i], declist[i]))

    source_dirs = os.listdir(dir)

    contained_count = 0
    uncontained_count = 0
    for src_dir in source_dirs:
        full_src_path = os.path.join("/mnt/DataDisk/Duncan/images4", src_dir)
        if os.path.isdir(full_src_path):
            if src_dir.__contains__('p'):
                ra, dec = src_dir.split('p')
                ra, dec = float(ra), float(dec)
            else:
                ra, dec = src_dir.split('m')
                dec = "-" + dec
                ra, dec = float(ra), float(dec)
            pos = (ra, dec)
            if positions.__contains__(pos):
                idx = positions.index(pos)
                label = classes[idx]
                print(label)
                contained_count += 1
                if label == 'GALAXY':
                    dest_folder = "/home/duncan/PycharmProjects/1XSDSS_DR16/data/sources/GALAXY"
                if label == 'QSO':
                    dest_folder = "/home/duncan/PycharmProjects/1XSDSS_DR16/data/sources/QSO"
                if label == 'STAR':
                    dest_folder = "/home/duncan/PycharmProjects/1XSDSS_DR16/data/sources/STAR"

                shutil.move(src=full_src_path, dst=dest_folder)

            else:
                print(f"can't find {pos} in table")
                uncontained_count += 1

    print(contained_count)
    print(uncontained_count)


# m = -2.5/ln(10) * [asinh((f/f0)/(2b)) + ln(b)].
# f0 in w1,w2,w3,w4:  306.681, 170.663, 29.0448, 8.2839  (Wright, Eisenhardt, etc, 2010)
# b for w1, w2: 0.1480469, 1.0154278
def cal_luptitude(w1flux, w2flux):  # shape: [16, 4]
    f0_w1 = 306.681
    f0_w2 = 170.663
    b_w1 = 0.148
    b_w2 = 1.015

    tmp1 = (w1flux / f0_w1) / (2 * b_w1)
    tmp2 = (w2flux / f0_w2) / (2 * b_w2)

    tmp3 = math.asinh(tmp1) + np.log(b_w1)
    tmp4 = math.asinh(tmp2) + np.log(b_w2)

    m1 = -2.5 / np.log(10) * tmp3
    m2 = -2.5 / np.log(10) * tmp4

    return m1, m2


def optimize_image(image_data):
    limg = np.arcsinh(image_data)
    limg = limg / limg.max()
    low, high = np.percentile(limg, (0.25, 99.5))
    opt_img = skie.rescale_intensity(limg, in_range=(low, high))  # skie 即 skimage.exposure
    return opt_img


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def get_sigma_clip(img, sigma=3, iters=100):
    """
    Do sigma clipping on the raw images to improve constrast of
    target regions.

    Reference
    =========
    [1] sigma clip
        http://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html
    """
    img_clip = sigma_clip(img, sigma=sigma, maxiters=iters)
    img_mask = img_clip.mask.astype(float)
    img_new = img * img_mask

    return img_new


def normalize(img):
    vmax = np.max(img)
    vmin = np.min(img)
    img = (img - vmin) / (vmax - vmin)
    return img


def CatPSimgMinMax(img_dat):
    medflux = np.median(img_dat)
    madflux = np.median(np.abs(img_dat - medflux))

    lomad, himad = (-2.0, 10.0)  # number of mads to the min and max

    minflux = medflux + lomad * madflux
    maxflux = medflux + himad * madflux

    # midflux = medflux  + 0.5 * (himad + lomad) * madflux
    # radflux = 0.5 * (himad - lomad) * madflux

    return minflux, maxflux


from astropy.visualization import ImageNormalize, SqrtStretch


def showImages(img):
    g = img[0]  # blue
    i = img[1]  # red
    r = img[2]  # green

    color_fits = torch.stack((i, r, g), dim=0)
    vmin, vmax = CatPSimgMinMax(color_fits)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    image_to_be_shown = color_fits.numpy().transpose((1, 2, 0))
    im = ax.imshow(image_to_be_shown, origin='lower', norm=norm)
    fig.colorbar(im)
    plt.show()


def have_source_in_center(image_dat):
    image_dat = image_dat[110:130, 110:130]
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3
    kenl = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kenl.normalize()

    threshold = detect_threshold(image_dat, nsigma=2.)
    segm = detect_sources(image_dat, threshold, npixels=5, kernel=kenl)
    if segm is None:
        return False
    else:
        return True
