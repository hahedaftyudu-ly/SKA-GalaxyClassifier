from astropy.table import Table

T = Table.read("data/SDSSxWISE_results_no_null.tbl", format="ipac")

ra_col = T['ra_01']
dec_col = T['dec_01']
class_col = T['class_01']

for i in range(len(ra_col)):
    ra_col[i] = round(ra_col[i], 3)

for j in range(len(dec_col)):
    dec_col[j] = round(dec_col[j], 3)

positions = list(zip(ra_col, dec_col))
sources_in_catalogue = list(zip(ra_col, dec_col, class_col))

# training_sets_path = "/mnt/DataDisk/Duncan/sources/STAR"
#
# training_set = os.listdir(training_sets_path)
# count = 0
#
# not_in_cat = []
# source_type_error = []
#
# for folder in training_set:
#     count += 1
#     print(count)
#     full_src_path = os.path.join(training_sets_path, folder)
#     if os.path.isdir(full_src_path):
#         if folder.__contains__('p'):
#             ra, dec = folder.split('p')
#         else:
#             ra, dec = folder.split('m')
#             dec = "-" + dec
#
#         ra_original, dec_original = float(ra), float(dec)
#         ra, dec = round(ra_original, 3), round(dec_original, 3)
#
#         the_source = (ra, dec)
#
#         if not positions.__contains__(the_source):
#             print(f"(ra, dec): {ra, dec}")
#             original_source = (ra_original, dec_original)
#             print(f"{original_source}(original radec) doesn't exist in catalogue")
#             not_in_cat.append(original_source)
#         else:
#             if not sources_in_catalogue.__contains__((ra, dec, "STAR")):
#                 source_type_error.append((ra_original, dec_original, "STAR"))


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
