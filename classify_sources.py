import os
import shutil

from astropy.table import Table

T = Table.read("data/DuncanSDSSdata.tbl", format='ipac')
ralist = list(T['ra'])
declist = list(T['dec'])
classes = list(T['class'])


def classify_downloaded_sources(dir):
    positions = []
    for i in range(len(ralist)):
        positions.append((ralist[i], declist[i]))

    source_dirs = os.listdir(dir)

    contained_count = 0
    uncontained_count = 0

    for src_dir in source_dirs:
        if src_dir == 'GALAXY':
            continue
        elif src_dir == 'QSO':
            continue
        elif src_dir == 'STAR':
            continue
        else:
            full_src_path = os.path.join(dir, src_dir)
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
                    dest_folder = None
                    if label == 'GALAXY':
                        dest_folder = os.path.join(dir, 'GALAXY')
                    if label == 'QSO':
                        dest_folder = os.path.join(dir, 'QSO')
                    if label == 'STAR':
                        dest_folder = os.path.join(dir, 'STAR')

                    shutil.move(src=full_src_path, dst=dest_folder)

                else:
                    print(f"can't find {pos} in table")
                    uncontained_count += 1

    print(contained_count)
    print(uncontained_count)


classify_downloaded_sources(dir='/mnt/DataDisk/Duncan/images70')
# Steps 1. classify  2. filter (the sources in sdss.specPhoto X WISE) 3. fix nan values


# images70  3
# images71  3
# images72  3
# images73  3
# images74  3
# images75  3
# images76  3
# images77  3
# images78  3
# images79  3
# images80  3
# images81  3
# images82  3
# images83  3
# images84  3
# images85  3
# images86  3
# images87  3
# images88  3
# images89  3
# images90  3
