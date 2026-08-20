import os

from astropy.table import Table

from download_data import getFits

T = Table.read("data/SDSS_Xmatch_QSO.csv", format="csv")
ra_col = T['ra']
dec_col = T['dec']
positions = list(zip(ra_col, dec_col))

QSO_dir = "/mnt/DataDisk/Duncan/qusars_newly_download"
old_QSO_dir = "data/sources/QSO"


def getDatasetsCoords(data_path):
    f_names = os.listdir(data_path)
    celes_bodies = []
    for c in f_names:
        if c.__contains__('p'):
            ra, dec = c.split('p')
        elif c.__contains__('m'):
            ra, dec = c.split('m')
            dec = '-' + dec
        else:
            raise ValueError
        ra, dec = float(ra), float(dec)
        ra, dec = round(ra, 5), round(dec, 6)
        celes_bodies.append((ra, dec))
    return celes_bodies


already_downloaded = getDatasetsCoords(QSO_dir)

qso_coords = getDatasetsCoords(old_QSO_dir)
print(f"qso training size: {len(qso_coords)}")
count = 0

for i in range(10000):
    print(f"i={i}")
    coord = positions[i]
    if already_downloaded.__contains__(coord):
        count += 1
        continue
    if not qso_coords.__contains__(coord):
        count += 1
        print(f"正在下载第{count}个新的类星体")
        ra = round(coord[0], 5)
        dec = round(coord[1], 6)
        if dec < 0:
            ra = str(ra)
            dec = str(-dec)
            dir_name = os.path.join(QSO_dir, f"{ra}m{dec}")
        else:
            ra = str(ra)
            dec = str(dec)
            dir_name = os.path.join(QSO_dir, f"{ra}p{dec}")
        fitsPath = getFits(ra, dec, dir_name)
