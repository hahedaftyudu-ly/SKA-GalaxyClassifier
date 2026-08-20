import os
import shutil
import numpy as np
import pandas as pd



if __name__ == '__main__':
    pd_p_RGZ = pd.read_csv('../data/catalogs/preprocessed_cat/PS_p_RGZ_samples.csv')
    pd_n_RGZ = pd.read_csv('../data/catalogs/preprocessed_cat/PS_n_RGZ_samples.csv')
