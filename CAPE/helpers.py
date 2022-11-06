import cv2
import numpy as np


def serialize_int64(obj):
    # Fix TypeError: Object of type int64 is not JSON serializable
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Type %s is not serializable" % type(obj))


def trim_recursive(frame):
    """
    https://stackoverflow.com/a/41670793
    :param frame:
    :return:
    """
    if frame.shape[0] == 0:
        return np.zeros((0, 0, 3))

    # crop top
    if not np.sum(frame[0]):
        return trim_recursive(frame[1:])
    # crop bottom
    elif not np.sum(frame[-1]):
        return trim_recursive(frame[:-1])
    # crop left
    elif not np.sum(frame[:, 0]):
        return trim_recursive(frame[:, 1:])
        # crop right
    elif not np.sum(frame[:, -1]):
        return trim_recursive(frame[:, :-1])
    return frame


def trim_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros((0, 0, 3))
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = frame[y:y + h, x:x + w]
    return crop


def trim_contours_exact(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros((0, 0, 3))
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = frame[y:y + h, x:x + w]
    return crop


def trim_iterative(frame):
    for start_y in range(1, frame.shape[0]):
        if np.sum(frame[:start_y]) > 0:
            start_y -= 1
            break
    if start_y == frame.shape[0]:
        if len(frame.shape) == 2:
            return np.zeros((0, 0))
        else:
            return np.zeros((0, 0, 0))
    for trim_bottom in range(1, frame.shape[0]):
        if np.sum(frame[-trim_bottom:]) > 0:
            break

    for start_x in range(1, frame.shape[1]):
        if np.sum(frame[:, :start_x]) > 0:
            start_x -= 1
            break
    for trim_right in range(1, frame.shape[1]):
        if np.sum(frame[:, -trim_right:]) > 0:
            break

    end_y = frame.shape[0] - trim_bottom + 1
    end_x = frame.shape[1] - trim_right + 1

    # print('iterative cropping x:{}, w:{}, y:{}, h:{}'.format(start_x, end_x - start_x, start_y, end_y - start_y))
    return frame[start_y:end_y, start_x:end_x]


def autocrop(image, threshold=0):
    """
    Crops
    any
    edges
    below or equal
    to
    threshold

    Crops
    blank
    image
    to
    1
    x1.

    Returns
    cropped
    image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image
