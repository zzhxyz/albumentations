import cv2
import numpy as np

from functools import wraps

from .utils import get_random_crop_coords, get_center_crop_coords
from albumentations.augmentations.keypoints_utils import angle_to_2pi_range


def angle_2pi_range(func):
    @wraps(func)
    def wrapped_function(keypoint, *args, **kwargs):
        (x, y, a, s) = func(keypoint, *args, **kwargs)
        return (x, y, angle_to_2pi_range(a), s)

    return wrapped_function


@angle_2pi_range
def keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols, **__):
    x, y, a, s, = keypoint[:4]
    height, width = rows, cols
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    angle = a + np.deg2rad(angle)
    scale = s * scale

    return x, y, angle, scale


@angle_2pi_range
def keypoint_vflip(keypoint, rows, *_, **__):
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Image height.
        cols( int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint
    angle = -angle
    return x, (rows - 1) - y, angle, scale


@angle_2pi_range
def keypoint_hflip(keypoint, rows, cols):
    """Flip a keypoint horizontally around the y-axis.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint
    angle = np.pi - angle
    return (cols - 1) - x, y, angle, scale


def keypoint_flip(keypoint, d, rows, cols):
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        d (int): Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        keypoint = keypoint_vflip(keypoint, rows, cols)
    elif d == 1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
    elif d == -1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
        keypoint = keypoint_vflip(keypoint, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return keypoint


@angle_2pi_range
def keypoint_rot90(keypoint, factor, rows, cols, **__):
    """Rotates a keypoint by 90 degrees CCW (see np.rot90)

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        factor (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if factor not in set {0, 1, 2, 3}

    """
    x, y, angle, scale = keypoint[:4]

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if factor == 1:
        x, y, angle = y, (cols - 1) - x, angle - np.pi / 2
    elif factor == 2:
        x, y, angle = (cols - 1) - x, (rows - 1) - y, angle - np.pi
    elif factor == 3:
        x, y, angle = (rows - 1) - y, x, angle + np.pi / 2

    return x, y, angle, scale


@angle_2pi_range
def keypoint_rotate(keypoint, angle, rows, cols, **__):
    """Rotate a keypoint by angle.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        angle (float): Rotation angle.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    matrix = cv2.getRotationMatrix2D(((cols - 1) * 0.5, (rows - 1) * 0.5), angle, 1.0)
    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return x, y, a + np.deg2rad(angle), s


def keypoint_scale(keypoint, scale_x, scale_y, **__):
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        scale_x (int): Scale coefficient x-axis.
        scale_y (int): Scale coefficient y-axis.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    return x * scale_x, y * scale_y, angle, scale * max(scale_x, scale_y)


def crop_keypoint_by_coords(keypoint, crop_coords, *_, **__):
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    x1, y1, x2, y2 = crop_coords
    return x - x1, y - y1, angle, scale


def keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols):
    """Keypoint random crop.

    Args:
        keypoint: (tuple): A keypoint `(x, y, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        h_start (int): Crop height start.
        w_start (int): Crop width start.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_keypoint_by_coords(keypoint, crop_coords, crop_height, crop_width, rows, cols)


def keypoint_center_crop(keypoint, crop_height, crop_width, rows, cols):
    """Keypoint center crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_keypoint_by_coords(keypoint, crop_coords, crop_height, crop_width, rows, cols)


def keypoint_transpose(keypoint):
    """Rotate a keypoint by angle.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]

    if angle <= np.pi:
        angle = np.pi - angle
    else:
        angle = 3 * np.pi - angle

    return y, x, angle, scale
