import os

src = "/mnt/DataDisk/Duncan/VLASS/components_v2"
VLASS = "/mnt/DataDisk/Duncan/Pan-STARRS_Big_Cutouts/VLASS_training_data"

imgs = os.listdir(VLASS)
count = 0
for img in imgs:
    arr = img.split("_")
    new_name = f"{arr[0]}.fits"
    os.rename(src=os.path.join(VLASS, img), dst=os.path.join(VLASS, new_name))
    count += 1
    print(count)

# source_dict = {}
# blocks = os.listdir(src)
# for block in blocks:
#     block_path = os.path.join(src, block)
#     imgs = os.listdir(block_path)
#     for img in imgs:
#         arrs = img.split("_")
#         name = arrs[0]
#         source_dict[name] = os.path.join(block_path, img)
#
#
# df = pd.read_csv("data/crossmatch/PS_positive_samples.csv")
# T = Table.from_pandas(df)
#
# df.set_index('VLASS_component_name', inplace=True)
#
# count = 0
# for row in T:
#     count += 1
#     print(count)
#     component_name = row['VLASS_component_name']
#     # already_copied = os.listdir(dst)
#     # if already_copied.__contains__(component_name):
#     #     print("already copied, skip")
#     #     continue
#     source_path = source_dict[component_name]
#     shutil.copy(source_path, dst)
