import os

from astropy.table import Table

filename = "SDSS_9col.tbl"

table = Table.read(filename, format='ipac')
ras = list(table['ra'])
decs = list(table['dec'])
zwarnings = list(table['zwarning'])

coords = list(zip(ras, decs))

zwarning_dict = {}

for i in range(len(coords)):
    zwarning_dict[coords[i]] = zwarnings[i]

gal_dir = "data/no_source_in_center/GALAXY"
star_dir = "data/no_source_in_center/STAR"
QSO_dir = "data/no_source_in_center/QSO"


def getDatasetsCoords(data_path):
    f_names = os.listdir(data_path)
    celes_bodies = []
    for c in f_names:
        if c.__contains__('p'):
            ra, dec = c.split('p')
            celes_bodies.append((float(ra), float(dec)))
        if c.__contains__('m'):
            ra, dec = c.split('m')
            dec = '-' + dec
            celes_bodies.append((float(ra), float(dec)))
        else:
            continue
    return celes_bodies


galaxy_coords = getDatasetsCoords(gal_dir)
star_coords = getDatasetsCoords(star_dir)
qso_coords = getDatasetsCoords(QSO_dir)

zw_0 = []
zw_5 = []
zw_16 = []
trainset_zwarning = []


def check_datasets_zwarning(dataset_coords):
    for i in range(len(dataset_coords)):
        key = dataset_coords[i]
        zwaring = zwarning_dict[key]
        trainset_zwarning.append(zwaring)
        if zwaring == 0:
            zw_0.append(key)
        elif zwaring == 16:
            zw_16.append(key)
        elif zwaring == 5:
            zw_5.append(key)
        else:
            pass


check_datasets_zwarning(galaxy_coords)
