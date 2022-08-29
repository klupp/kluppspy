import numpy as np
import cv2
from kluppspy.cv import utils as cvu
from kluppspy.devices import calculator as calc


def add_border(im, border_width=None, left_sample_loc=(5, 5), right_sample_loc=(5, -5)):
    """
    Add white border to `Vaillant VRT 300`

    add white borders left and right in order to contain the screen inside the picture.

    @param im of the thermostat on which the border will be added.
    @param border_width the width of the border to be added.
    @param left_sample_loc sample pixel location for the left border.
    @param right_sample_loc sample pixel location for the right border.
    """

    # if the border width is not defined make it one tenth of the width of the picture.
    if border_width is None:
        border_width = int(im.shape[1] / 10)

    # We take the samples of the image determined by the sample loc parameters.
    h, w = left_sample_loc
    left_sample = tuple([int(val) for val in im[h, w]])

    h, w = right_sample_loc
    right_sample = tuple([int(val) for val in im[h, w]])

    # Replace the left most and right columns with the sample values (Ensures monotone white color of the end pixels)
    im[0:400, 0:1] = left_sample
    im[0:300, -1:] = right_sample

    # We blur the left and right border in order to have smooth transitions.
    # We remove the artificial edges created by the replacement.
    im[0:1450, 0:1] = cv2.blur(im[0:1450, 0:1], (1, 400))
    im[0:1500, -1:] = cv2.blur(im[0:1500, -1:], (1, 400))

    # Finally we extend both sides by the number of pixels specified in the border_width
    im = cv2.copyMakeBorder(im, border_width, border_width, border_width, border_width, cv2.BORDER_REPLICATE, None)
    return im, border_width


def preprocess_image(im):
    im, border_pixels = add_border(im)
    # Scale down the image so we can find the screen faster (it is big enough)
    ratio = 300.0 / im.shape[0]
    im = cvu.scale_im(im.copy(), ratio)

    return im, ratio, border_pixels


def invert_changes(points, scale, border_width):
    return (np.array(points) // scale).astype(np.int32) - border_width


def find_screen(im):
    img, scale, border_width = preprocess_image(im)
    screen_cnt, cnts = cvu.find_sqare(img)
    cnt = None
    if screen_cnt is not None:
        cnt = invert_changes(screen_cnt, scale, border_width)
    cnts = [invert_changes for cnt in cnts]
    return cnt, cnts


def extract_screen(im):
    screen_cnts, cnts = find_screen(im)
    screen_pts = screen_cnts[:, 0, :]
    return cvu.four_point_transform(im, screen_pts)


def get_regions(im):
    """
    Get Regions of Interest boundaries from thermostat screen Valliant.
    Please not that you will have to provide screen of size `1000x700`
    because the boundaries are set manually for the given thermostat model.

    @param im thermostat screen
    """
    regions = {
        "temperature": cvu.to_rect(85, 250, 525, 445),
        "power": cvu.to_rect(700, 130, 830, 250),
        "auto_on": cvu.to_rect(80, 130, 190, 240),
        "day_on": cvu.to_rect(260, 130, 360, 240),
        "night_on": cvu.to_rect(350, 130, 450, 240),
    }
    return regions


def extract_regions(im):
    """
    Extract regions of interest from a Valliant thermostat return them in a dictionary.
    The values are images of the regions while the keys are (`temperature`, `power`, `day_on`, `night_on`, `auto_on`)

    @param im image of a Valliant thermostat
    """
    screen_im = extract_screen(im)
    screen_im = cv2.resize(screen_im, (1000, 700))
    regions = get_regions(screen_im)

    region_ims = {}
    for name, cnt in regions.items():
        region = cvu.four_point_transform(screen_im, cnt[:, 0, :])
        region_ims[name] = region
    return region_ims


def extract_temperature_digits(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # #     im = cvu.scale_im(im, scale=0.3)
    #     print(im.shape)

    # Histogram
    #     clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    #     im = clahe.apply(im)

    #     im = cv2.equalizeHist(im)

    #     im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    # 	cv2.THRESH_BINARY_INV, 65, 65)
    #     im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = np.ones((35, 35), np.uint8)
    im = 255 - im

    im = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, kernel)
    _, im = cv2.threshold(im, 8, 255, cv2.THRESH_BINARY)
    im = cv2.copyMakeBorder(im, 0, 0, 0, 40, cv2.BORDER_CONSTANT, None, 0)

    # Remove itallic from the font
    M = np.float32([[1, 0.28, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    im = cv2.warpPerspective(im, M, (im.shape[1], im.shape[0]))
    im[150:190, 300:335] = 0
    kernel = np.ones((8, 8), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((20, 20), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

    #     plt.imshow(im, cmap='gray')

    cnts, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda cnt: cnt[0, 0, 0], reverse=False)

    digits = {}
    for (i, c) in enumerate(cnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        hull = cv2.convexHull(c, False)
        peri = cv2.arcLength(hull, True)
        if peri < 300:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        digit = im[y:y + h, x:x + w]
        #         roi = cv2.resize(roi, (57, 88))
        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = digit

    new_digits = {}
    for i in range(len(digits)):
        digit = digits[i]
        border = digits[2].shape[1] - digit.shape[1]
        if border > 10:
            digit = cv2.copyMakeBorder(digit, 0, 0, border, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        kernel = np.ones((10, 10), np.uint8)
        #         font = cv2.morphologyEx(digit, cv2.MORPH_TOPHAT, kernel)
        digit = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel)
        new_digits[i] = cv2.resize(digit, (57, 88))
    return new_digits


def get_temperature(im):
    template_digits = calc.extract_digits()
    digits = extract_temperature_digits(im)
    groupOutput = []
    for digit in digits.values():
        scores = []
        for template in template_digits.values():
            # apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(digit, template, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # the classification for the digit ROI will be the reference
        # digit name with the *largest* template matching score
        groupOutput.append(str(np.argmax(scores)))
    if len(groupOutput) == 3:
        return float(''.join(groupOutput)) / 10
    return None


