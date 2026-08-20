import os
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn import metrics
from skimage import filters, measure, color, data, graph, morphology
from skimage.filters.thresholding import threshold_multiotsu
from scipy import ndimage as ndi

from utils import get_sigma_clip, remove_nan, normalize

retina_source = data.retina()

_, ax = plt.subplots()
ax.imshow(retina_source)
ax.set_axis_off()
_ = ax.set_title('Human retina')
plt.show()



