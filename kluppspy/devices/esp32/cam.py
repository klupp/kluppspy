import cv2


def imread(path):
    """
    Read and the unwanted issues from the esp32 cam and yield a clear picture.
    """
    im = cv2.imread(path)
    return fix(im)


def fix(im):
    """
    Fix the unwanted issues from the esp32 cam and yield a clear picture.
    """
    # Since the image is rotated to the left by 90 degrees
    # we first rotate it by 90 degrees right (clockwise) to get the up standing image.
    # This can be removed if I fix the camera
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    return im