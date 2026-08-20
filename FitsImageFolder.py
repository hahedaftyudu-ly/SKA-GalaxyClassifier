import os
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import find_classes

from utils import remove_nan, cal_luptitude, CatPSimgMinMax, optimize_image

# --- WISE 星等查表 ---
# 模型 A 的输入除 5 波段图像外，还需要每个源的 2 个 WISE asinh 星等（用于区分恒星/AGN）。
# 训练时用的原始星表是 SDSSxWISE_cat.tbl（CatWISE2020，IPAC 格式，原在 Expansion/catalogues/ 下）；
# data/catalogs/SDSS_clean_cat_Duncan.csv 是它的 CSV 版（同样的 source_id/w1flux/w2flux，读取快得多）。
# 默认用 data/ 下的 CSV；要用原始 .tbl 就把 WISE_CATALOGUE 指过去（.tbl 走 IPAC、较慢）。
# 首次用到时惰性加载并缓存，避免 import 即读大文件。
WISE_CATALOGUE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "catalogs", "SDSS_clean_cat_Duncan.csv")
_WISE_CACHE = None


def _wise_lookup():
    global _WISE_CACHE
    if _WISE_CACHE is None:
        if WISE_CATALOGUE.endswith(".tbl"):
            from astropy.table import Table
            t = Table.read(WISE_CATALOGUE, format="ipac")
            rows = zip(t["source_id"], t["w1flux"], t["w2flux"])
        else:
            import pandas as pd
            df = pd.read_csv(WISE_CATALOGUE, usecols=["source_id", "w1flux", "w2flux"])
            rows = zip(df["source_id"], df["w1flux"], df["w2flux"])
        _WISE_CACHE = {str(sid): (float(w1), float(w2)) for sid, w1, w2 in rows}
    return _WISE_CACHE


def _parse_radec(fits_filename):
    """从 'stack_g_ra359.364420_dec-1.131493_arcsec60_...' 解析 (ra, dec)，仅用于结果记录。"""
    import re
    m = re.search(r"ra(-?\d+\.?\d*)_dec(-?\d+\.?\d*)", fits_filename)
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


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
            allow_empty: bool = False,
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
    def __fits_loader(fits_name):
        src_dir_contents = sorted(os.listdir(fits_name))
        short_name = fits_name.split(os.path.sep)[-1]  # = CatWISE source_id（即文件夹名）

        # WISE 星等：按 source_id 查 w1flux/w2flux，经 cal_luptitude 转 2 个 asinh 星等。
        # （恢复此前被注释掉的逻辑——OptDataSet 和 model(image, wise) 都依赖它。）
        w1flux, w2flux = _wise_lookup().get(short_name, (0.0, 0.0))
        wise_magnitude_info = np.asarray(cal_luptitude(w1flux, w2flux))

        # 5 波段图像立方体
        img_list = []
        for i in range(5):
            img_path = os.path.join(fits_name, src_dir_contents[i])
            img_list.append(remove_nan(fits.getdata(img_path)))
        img_dat = optimize_image(np.stack(img_list, axis=2))

        # 源坐标（从 fits 文件名解析，仅用于结果记录）
        ps_ra, ps_dec = _parse_radec(src_dir_contents[0])

        return img_dat, wise_magnitude_info, (ps_ra, ps_dec)
