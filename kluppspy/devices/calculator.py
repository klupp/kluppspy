import numpy as np
import cv2
from kluppspy.cv import utils as cvu


def get_digits_im():
    digits_path = '../images/fonts/calculator_numbers.png'
    im = cv2.imread(digits_path)
    im = cv2.copyMakeBorder(im, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, (255, 255, 255))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def extract_digits_from_im(im):
    font = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((8, 8), np.uint8)
    # font = cv2.morphologyEx(font, cv2.MORPH_TOPHAT, kernel)
    font = cv2.morphologyEx(font, cv2.MORPH_CLOSE, kernel)

    font_cnts, _ = cv2.findContours(font.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    font_cnts = sorted(font_cnts, key=lambda cnt: cnt[0, 0, 0], reverse=False)

    digits = {}
    for (i, c) in enumerate(font_cnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        roi = font[y:y + h, x:x + w]
        #         roi = cv2.resize(roi, (57, 88))
        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi

    border = digits[0].shape[1] - digits[1].shape[1]
    digits[1] = cv2.copyMakeBorder(digits[1], 0, 0, border, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    new_digits = {}
    for i in range(len(digits)):
        digit = digits[i]
        kernel = np.ones((10, 10), np.uint8)
        #         font = cv2.morphologyEx(digit, cv2.MORPH_TOPHAT, kernel)
        font = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel)
        new_digits[i] = cv2.resize(digit, (57, 88))
    return new_digits


def extract_digits():
    im = get_digits_im()
    return extract_digits_from_im(im)
