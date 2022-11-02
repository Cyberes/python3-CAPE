import json
import ntpath
import os

from .panelextractor import *

"""
This is the main file and holds all the functions a user will interact with.
"""


def serialize_int64(obj):
    # Fix TypeError: Object of type int64 is not JSON serializable
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Type %s is not serializable" % type(obj))


def process_comic_panel(comic_panel_path):
    """
    Processes a single comic panel and places
    the processed panels information in dest folder.

    @param comic_panel_path to the comic panel file.
    """

    image = cv2.imread(comic_panel_path)

    comic_panels = find_comic_panels(image)

    for panelInfo in comic_panels:
        x, y, w, h = panelInfo[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # cv2.imshow("Output", image)
    # cv2.waitKey(0)

    out_path = "res_" + ntpath.basename(comic_panel_path)

    filename = os.path.basename(comic_panel_path)
    filename_not_ext = os.path.splitext(filename)[0]

    metadata_dest = os.path.splitext(comic_panel_path)[0] + ".cpanel"

    panel_metadata = generate_panel_metadata(comic_panels)

    # Add image file name to improve linkage to the source image
    panel_metadata['imagePath'] = filename

    panel_metadata_json = json.dumps(panel_metadata, default=serialize_int64)
    metadata_file = open(metadata_dest, "w")
    metadata_file.write(panel_metadata_json)

    print(out_path)

    if debug:
        cv2.imwrite(out_path, image)

    return panel_metadata


def process_comic_panels_from_dir(comic_panel_dir):
    """
    Processes every single comic page in the given directory.
    @param comic_panel_dir - path of the source directory
    """
    for filename in os.listdir(comic_panel_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(filename)
            process_comic_panel(os.path.join(comic_panel_dir, filename))


def crop_panels_from_dir(metadata_dir_path, dest_dir):
    """
    Function to extract all the panels from every comic page in the
    given directory. This will load the metadata file for the
    panels recognized using the panel recognizer and extract
    each of them into its own image file
    """

    # for root, dirs, files in os.walk(metadataPathDir):
    for file in os.listdir(metadata_dir_path):
        file_path = os.path.join(metadata_dir_path, file)
        if not os.path.isfile(file_path):
            continue
        # absolutePath = os.path.abspath(file)
        filename, ext = os.path.splitext(file)
        # print(filename + " " + ext)

        if ext == '.cpanel':
            print(filename + " " + ext)
            print(file_path)
            crop_panels(file_path, dest_dir)
            pass


def crop_panels(metadata_path, dest_dir):
    """
    Crops panels and saves them on the destination
    """
    # TODO Load metadata file
    with open(metadata_path) as jsonFile:
        metadata = json.load(jsonFile)

    image_dir = os.path.dirname(metadata_path)
    print(metadata)
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
        print(out_path)
        cv2.imwrite(out_path, panel_img)
    pass


def page(path, dest_path):
    """
    Analyze and crop a single page.
    :param dest_path:
    :param path: Path to an image file.
    :return:
    """
    if not os.path.isfile(path):
        raise Exception(f'Not a file: {path}')
    image_path = process_comic_panel(path)['imagePath']
    crop_panels(os.path.join(os.path.dirname(path), os.path.splitext(image_path)[0] + '.cpanel'), dest_path)


def directory(dir_path, dest_path):
    """
    Analyze and crop all images in a directory.
    :param path:
    :param dest_path:
    :return:
    """
    if not os.path.isdir(dir_path):
        raise Exception(f'Not a directory: {dir_path}')
    for file in os.listdir(dir_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            page(os.path.join(dir_path, file), dest_path)
