import json
import ntpath
import os

from .helpers import *
from .panelextractor import *

"""
This is the main file and holds all the functions a user will interact with.
"""


def crop_black_borders(file_path, out_path):
    """
    EXPERIMENTAL
    Crop the black borders from an image file.
    :param file_path:
    :param out_path:
    :return:
    """
    img = cv2.imread(file_path)
    trim = trim_recursive(img)
    cv2.imwrite(out_path, trim)


def metadata(comic_panel_path, save_path=False):
    """
    Process a single comic panel
    :param comic_panel_path:
    :param save_path:
    :return:
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
    :param image_path:
    :param output_dir:
    :param save_metadata:
    :return:
    """
    if not os.path.isfile(image_path):
        raise Exception(f'Not a file: {image_path}')

    mdata = metadata(image_path, save_metadata)

    # image_dir = image_path
    # image_filename = ''

    if mdata['version'] > 1:
        image_filename = mdata['imagePath']
    else:
        # TODO Get image from path ?
        pass

    loaded_image = cv2.imread(image_path)

    # For every panel crop panels
    panels = mdata['panels']
    for index, panel in enumerate(panels):
        box = panel['box']
        x = int(box['x'])
        y = int(box['y'])
        w = int(box['w'])
        h = int(box['h'])
        # print(box)
        panel_img = loaded_image[y: y + h, x: x + w]

        # Save every panel under filename_panelIndex.imageExtension
        filename, ext = os.path.splitext(os.path.split(image_path)[-1])
        out_file = filename + '_' + str(index).zfill(3) + ext

        if not os.path.isdir(output_dir):
            raise Exception(f'Output directory does not exist: {output_dir}')

        out_path = os.path.join(output_dir, out_file)
        cv2.imwrite(out_path, panel_img)

        cropped_filename, cropped_ext = os.path.splitext(os.path.split(out_file)[-1])
        cropped_out_path = os.path.join(output_dir, f'{cropped_filename}_cropped{cropped_ext}')

        # crop_black_borders(out_path, cropped_out_path)
    return len(panels)
