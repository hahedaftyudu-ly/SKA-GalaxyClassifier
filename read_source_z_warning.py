import os

from astropy.table import Table

PATH = 'data/too_many_nan_values/STAR'
T = Table.read("SDSS_9col.tbl", format="ipac")
ra_list = list(T['ra'])
dec_list = list(T['dec'])
z_warnings = list(T['zwarning'])
positions = list(zip(ra_list, dec_list))

dirs = os.listdir(PATH)
should_exclude = []
for dir in dirs:
    if dir.__contains__('m'):
        ra = dir.split('m')[0]
        dec = "-" + dir.split('m')[1]
    elif dir.__contains__('p'):
        ra = dir.split('p')[0]
        dec = dir.split('p')[1]
    else:
        raise IOError

    ra = float(ra)
    dec = float(dec)
    idx = positions.index((ra, dec))
    if not (z_warnings[idx] == 0 or z_warnings == 16):
        should_exclude.append(dir)

    # if not positions.__contains__((ra, dec)):
    #     print((ra, dec))
    #     should_exclude.append((ra, dec))
