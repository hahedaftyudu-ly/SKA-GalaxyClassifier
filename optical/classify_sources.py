import os
import shutil

import astropy.table


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
        if src_dir == 'GALAXY':
            continue
        if src_dir == 'QSO':
            continue
        if src_dir == 'STAR':
            continue

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


classify_downloaded_sources(dir='/mnt/DataDisk/Duncan/images44')
