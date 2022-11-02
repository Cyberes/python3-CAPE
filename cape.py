import os
import pathlib
import sys

import CAPE as cape

pathlib.Path(os.path.join(sys.path[0], 'test')).mkdir(parents=True, exist_ok=True)

cape.page(os.path.join(sys.path[0], 'sample-image.jpg'), os.path.join(sys.path[0], 'test'))
# cape.directory('Sources', 'test/bulk')
