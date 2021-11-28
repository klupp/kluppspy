import numpy as np
import cv2
from kluppspy.cv import utils as cvu


def add_border(im):
    border_color = tuple([int(val) for val in im[5, 5]])
    im[0:400, 0:1] = border_color
    im[0:1450, 0:1] = cv2.blur(im[0:1450, 0:1], (1, 400))
    border_color = tuple([int(val) for val in im[5, -5]])
    im[0:300, -1:] = border_color
    im[0:1500, -1:] = cv2.blur(im[0:1500, -1:], (1, 400))
    border_pixels = 100
    # im = cv2.copyMakeBorder(im, 0, 0, 20, 20, cv2.BORDER_CONSTANT, None, border_color)
    im = cv2.copyMakeBorder(im, border_pixels, border_pixels, border_pixels, border_pixels, cv2.BORDER_REPLICATE, None)
    return im, border_pixels


def preprocess_image(im):
    im, border_pixels = addBorder(im)
    # Scale down the image so we can find the screen faster (it is big enough)
    ratio = 300.0 / im.shape[0]
    im = cvu.scale_im(im.copy(), ratio)

    return im, ratio, border_pixels


def invert_changes(points, scale, border_pixels):
    return (np.array(points) // scale).astype(np.int32) - border_pixels