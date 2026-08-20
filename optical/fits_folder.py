import os
import logging
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
from astropy.io import fits
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import find_classes

from tools.utils import remove_nan, CatPSimgMinMax

# T = Table.read('SDSS_clean_cat_Duncan.tbl', format='ipac')
# ra_col = T['ra']
# dec_col = T['dec']
# w1flux = T['w1flux']
# w2flux = T['w2flux']
# w1sky = T['w1sky']
# w2sky = T['w2sky']
# positions = list(zip(ra_col, dec_col))
# fluxes = list(zip(w1flux, w2flux, w1sky, w2sky))


def make_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    instances = []
    available_classes = set()

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, dirs, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for idx, dir in enumerate(sorted(dirs)):
                path = os.path.join(root, dir)
                if os.path.isdir(path):
                    item = path, class_index
                    instances.append(item)
                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


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
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    @staticmethod
    def __fits_loader(dir):
        src_dir_contents = os.listdir(dir)
        if len(src_dir_contents) != 5:
            logging.error(f"{dir} must contain 5 fits files, current is {len(src_dir_contents)}")
            raise IOError
        src_dir_contents.sort()

        paths = dir.split(os.path.sep)
        short_name = paths[-1]

        if short_name.__contains__('m'):
            radec = short_name.split('m')
            ra = float(radec[0])
            dec = float("-" + radec[1])

        else:
            radec = short_name.split('p')
            ra = float(radec[0])
            dec = float(radec[1])



        img_list = []
        image_cube_vmax = 0
        for i in range(5):
            fits_f = src_dir_contents[i]
            single_channel_img_dat = fits.getdata(os.path.join(dir, fits_f))

            ################ Step 1. remove nan or np.inf by interplating new value ######################
            single_channel_img_dat = remove_nan(single_channel_img_dat)

            ################ Step 2. CatPSMinMax ######################
            center_region = single_channel_img_dat[100:140, 100:140]
            _, vmax = CatPSimgMinMax(center_region)
            single_channel_img_dat = np.where(single_channel_img_dat > vmax, vmax, single_channel_img_dat)
            if vmax > image_cube_vmax:
                image_cube_vmax = vmax

            img_list.append(single_channel_img_dat)

        img_dat = np.stack(img_list, axis=2)  # img_dat.shape: (240, 240, 5)

        ############### Step 3. ZScale stretch ######################
        # img_dat = LinearStretch().__call__(ZScaleInterval().__call__(img_dat))

        return img_dat
