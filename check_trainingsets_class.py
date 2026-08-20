import os

from astropy.table import Table

T = Table.read("data/catalogs/SDSS_clean_cat/SDSSxWISE_cat.tbl", format="ipac")

ra_col = T['ra_01']
dec_col = T['dec_01']
class_col = T['class_01']
source_id_col = T['source_id']

positions = list(zip(ra_col, dec_col))
sources_in_catalogue = list(zip(ra_col, dec_col, class_col))

training_sets_path = "/mnt/DataDisk/Duncan/sources/STAR"

training_set = os.listdir(training_sets_path)
count = 0

not_in_cat = []
source_type_error = []

for folder in training_set:
    count += 1
    print(count)
    full_src_path = os.path.join(training_sets_path, folder)
    if os.path.isdir(full_src_path):
        if folder.__contains__("_") and folder.__contains__("-"):  # already renamed
            continue

        if folder.__contains__('p') and not folder.__contains__("_"):
            ra, dec = folder.split('p')
        else:
            ra, dec = folder.split('m')
            dec = "-" + dec

        ra_original, dec_original = float(ra), float(dec)
        ra, dec = round(ra_original, 5), round(dec_original, 6)

        the_source = (ra, dec)

        if not positions.__contains__(the_source):
            print(f"(ra, dec): {ra, dec}")
            original_source = (ra_original, dec_original)
            print(f"{original_source}(original radec) doesn't exist in catalogue")
            raise ValueError

        else:
            idx = positions.index(the_source)
            source_id = source_id_col[idx]
            new_name = os.path.join(training_sets_path, source_id)
            os.rename(full_src_path, new_name)

# for coords in not_in_cat:
#     ra = coords[0]
#     dec = coords[1]
#     if dec < 0:
#         dec = -dec
#         src_dir_name = str(ra) + 'm' + str(dec)
#     else:
#         src_dir_name = str(ra) + 'p' + str(dec)
#     full_path = os.path.join('/mnt/DataDisk/Duncan/sources/QSO', src_dir_name)
#     shutil.rmtree(full_path)
