import os
import shutil

import numpy as np
from astropy.io import fits
import os
import shutil

import numpy as np
from astropy.io import fits

clean_star_dir_path = "/mnt/DataDisk/Duncan/sources/STAR"
clean_quasar_dir_path = "/mnt/DataDisk/Duncan/sources/QSO"


def removeNAN(source_path):
    print("star remove")
    count = 0
    for root, dirs, fnames in sorted(os.walk(source_path, followlinks=True)):  # here
        for dir in sorted(dirs):
            count += 1
            print(count)
            src_path = os.path.join(root, dir)
            fits_img = [os.path.join(src_path, f) for f in os.listdir(src_path)]

            for f in fits_img:
                try:
                    img_dat = fits.getdata(f)
                    x, y = np.where(np.isnan(img_dat))
                    if len(x) > 100:  # drop this sources
                        print(f"{f} contains {len(x)} nan pixels, remove this source")
                        shutil.rmtree(src_path)
                        break
                except TypeError as e:
                    print(e)
                    print(f"image file name is {f}")
                    break
