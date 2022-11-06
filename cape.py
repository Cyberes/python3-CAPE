import glob
import os
import pathlib
import sys

import CAPE as cape

pathlib.Path(os.path.join(sys.path[0], 'test-output')).mkdir(parents=True, exist_ok=True)

# cape.crop_page(os.path.join(sys.path[0], 'sample1.jpg'), os.path.join(sys.path[0], 'test'))
# cape.crop_page(os.path.join(sys.path[0], 'sample-image.jpg'), os.path.join(sys.path[0], 'test'))

for file in glob.glob(f'{os.path.join(sys.path[0], "test-images")}/*.jpg'):
    print(file)
    cape.crop_page(os.path.join(sys.path[0], 'test-images', file), os.path.join(sys.path[0], 'test-output'))
