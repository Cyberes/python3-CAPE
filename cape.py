import os
import pathlib
import sys

import CAPE as cape

pathlib.Path(os.path.join(sys.path[0], 'test')).mkdir(parents=True, exist_ok=True)

cape.crop_page(os.path.join(sys.path[0], 'sample-image.jpg'), os.path.join(sys.path[0], 'test'))

# if not os.path.isdir(dir_path):
#     raise Exception(f'Not a directory: {dir_path}')
# for file in os.listdir(dir_path):
#     if file.endswith(".png") or file.endswith(".jpg"):
#         page(os.path.join(dir_path, file), dest_path)
