import json
import ntpath
import os

from .helpers import *
from .panelextractor import *

"""
This is the main file and holds all the functions a user will interact with.
"""


def metadata(comic_panel_path, save_path=False):
    """
    Processes a single comic panel
    @param comic_panel_path to the comic panel file.
    """

    image = cv2.imread(comic_panel_path)

    comic_panels = find_comic_panels(image)

    for panelInfo in comic_panels:
        x, y, w, h = panelInfo[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

    out_path = "res_" + ntpath.basename(comic_panel_path)

    filename = os.path.basename(comic_panel_path)
    filename_not_ext = os.path.splitext(filename)[0]

    metadata_dest = os.path.splitext(comic_panel_path)[0] + ".cpanel"

    panel_metadata = generate_panel_metadata(comic_panels)

    panel_metadata['imagePath'] = filename

    if save_path is not False:
        if os.path.isdir(save_path):
            panel_metadata_json = json.dumps(panel_metadata, default=serialize_int64)
            metadata_file = open(metadata_dest, "w")
            metadata_file.write(panel_metadata_json)
        else:
            raise Exception(f'Not a directory: {save_path}')
    return panel_metadata


def crop_page(image_path, output_dir, save_metadata=False):
    """
    Analyze and crop a single page.
    :param output_dir:
    :param path: Path to an image file.
    :return:
    """
    if not os.path.isfile(image_path):
        raise Exception(f'Not a file: {image_path}')

    metadata = generate_metadata(image_path, save_metadata)

    image_dir = image_path
    image_filename = ''

    if metadata['version'] > 1:
        image_filename = metadata['imagePath']
    else:
        # TODO Get image from path ?
        pass

    loaded_image = cv2.imread(os.path.join(image_dir, image_filename))

    # For every panel crop panels
    panels = metadata['panels']
    for index, panel in enumerate(panels):
        box = panel['box']
        x = int(box['x'])
        y = int(box['y'])
        w = int(box['w'])
        h = int(box['h'])
        print(box)
        panel_img = loaded_image[y: y + h, x: x + w]
        if debug:
            cv2.imshow("Panel", panel_img)
            cv2.waitKey(0)
        # Save every panel under filename_panelIndex.imageExtension
        filename, ext = os.path.splitext(image_filename)
        out_file = filename + '_' + str(index).zfill(3) + ext

        if not os.path.isdir(dest_dir):
            raise Exception(f'Output directory does not exist: {dest_dir}')

        out_path = os.path.join(dest_dir, out_file)
        cv2.imwrite(out_path, panel_img)
    # pass
    return len(panels)
