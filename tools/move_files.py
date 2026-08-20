import os
import shutil

root = "/mnt/DataDisk/Duncan/images11/GALAXY"

src_dirs = os.listdir(root)
fits_f = []

for src_dir in src_dirs:
    src_dir_path = os.path.join(root, src_dir)
    if os.path.isdir(src_dir_path):

        contents = os.listdir(src_dir_path)
        if len(contents) == 0:
            shutil.rmtree(src_dir_path)
    # t = [os.path.join(root, src_dir, c) for c in contents]
    # for f in t:
    #     shutil.move(f, root)


print("lou")