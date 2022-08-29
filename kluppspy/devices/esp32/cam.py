import cv2
import kluppspy.cv.utils as cvu


def fix(im):
    """
    Fix the unwanted issues from the esp32 cam and yield a clear picture.
    """
    # Since the image is rotated to the left by 90 degrees
    # we first rotate it by 90 degrees right (clockwise) to get the up standing image.
    # This can be removed if I fix the camera
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    return im


def imread(path):
    """
    Read and the unwanted issues from the esp32 cam and yield a clear picture.
    """
    im = cv2.imread(path)
    return fix(im)


def dir_images(path):
    """
    Read all images
    """
    image_paths = cvu.dir_image_paths(path)
    for image_path in image_paths:
        yield imread(image_path)
