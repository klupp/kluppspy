import glob
import os

import numpy as np
import cv2
import matplotlib.colors as colors
import matplotlib.cm as cm
from kluppspy.cv import image

######################## UTILS

def empty_kernel():
    vect = np.array([0, 1, 0])
    return vect.reshape(1, -1)


def avg_kernel():
    return (np.ones((1, 3)) / 3).reshape(1, -1)


def binomial_kernel():
    vect = np.array([1, 2, 1]) / 4
    return vect.reshape(1, -1)


def gauss(x, sigma=1, mu=0):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def gaussdx(x, sigma=1, mu=0):
    return -((x - mu) / (np.sqrt(2 * np.pi) * sigma ** 3)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


######################## IM UTILS


def create_mask(im, gray_mask):
    im = im.astype(np.float32)
    gray_mask = gray_mask.astype(np.float32)
    mask255 = (gray_mask * 255) * 0.7

    # Create a mask
    mask = cv2.cvtColor(mask255.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Calculate image with red mask
    masked = im.copy()
    masked[:, :, 2] += mask255
    # Normalize the too high mask values
    masked[masked > 255] = 255

    return mask.astype(np.float32), masked


def calc_mask(gray_mask):
    gray_mask = gray_mask.astype(np.float32)
    mask255 = (gray_mask * 255) * 0.7

    # Create a mask
    mask = cv2.cvtColor(mask255.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return mask


def scale_im(img, scale=1):
    if scale == 1:
        return img
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def plot_image_terain(im, ax, perserve_ratio=True):
    if perserve_ratio:
        lim = np.max(im.shape)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        min_val = np.min(im)
        max_val = np.max(im)
        ax.set_zlim(min_val * 2, max_val * 2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    h, w = im.shape

    x_pos, y_pos = np.meshgrid(np.arange(w), np.arange(h))

    z_pos = np.zeros_like(im).reshape(-1)
    x_size = np.ones_like(x_pos).reshape(-1)
    y_size = np.ones_like(y_pos).reshape(-1)
    z_size = im.reshape(-1)
    x_pos = x_pos.reshape(-1)
    y_pos = y_pos.reshape(-1)

    norm = colors.Normalize(z_size.min(), z_size.max())
    clrs = cm.autumn(norm(z_size))
    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color=clrs, alpha=0.6)


def plot_image_surface(im, ax, perserve_ratio=True):
    if perserve_ratio:
        lim = np.max(im.shape)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        min_val = np.min(im)
        max_val = np.max(im)
        ax.set_zlim(min_val * 2, max_val * 2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    h, w = im.shape

    x_pos, y_pos = np.meshgrid(np.arange(w), np.arange(h))

    #     z_pos = np.zeros_like(im).reshape(-1)
    #     x_size = np.ones_like(x_pos).reshape(-1)
    #     y_size = np.ones_like(y_pos).reshape(-1)
    #     z_size = im.reshape(-1)
    #     x_pos = x_pos.reshape(-1)
    #     y_pos = y_pos.reshape(-1)

    norm = colors.Normalize(im.min(), im.max())
    clrs = cm.autumn(norm(im))
    ax.plot_surface(x_pos, y_pos, im, cmap='autumn', alpha=0.8)


######################## Filters


##### Point-Filters


def clip(im, a=0, b=254):
    res_im = im.copy()
    res_im[res_im <= a] = a
    res_im[res_im >= b] = b
    return res_im


def norm(im, a_old=None, b_old=None, a_new=0, b_new=255):
    if a_old is None:
        a_old = np.min(im)
    if b_old is None:
        b_old = np.max(im)
    return clip((im - a_old) / (b_old - a_old), 0, 1) * (b_new - a_new) + a_new


def threshold(im, threshold=127):
    im = norm(im)
    return (im > threshold).astype(np.float32) * 255


class Thresholding:
    def __init__(self, param_im):
        self.im = param_im

    def get_trheshold_isodata(self):
        intensity_sup = np.max(self.im)

        param_im_histogram = cvu.Histogram(self.im)
        intensities, param_im_hist = param_im_histogram.get_hist()

        intensities, param_im_hist = intensities[:intensity_sup + 1], param_im_hist[:intensity_sup + 1]

        hist_intensities = param_im_hist * intensities
        param_im_integral = np.cumsum(hist_intensities)
        param_im_inv_integral = np.flip(np.cumsum(np.flip(hist_intensities)))

        param_im_cum_hist = np.cumsum(param_im_hist)
        param_im_cum_inv_hist = np.flip(np.cumsum(np.flip(param_im_hist)))

        phi_a = param_im_integral / param_im_cum_hist
        phi_b = param_im_inv_integral / param_im_cum_inv_hist
        phi = (phi_a + phi_b) / 2

        return (intensities, phi_a, phi_b, phi)

    def threshold(self):
        intensity_inf = np.min(self.im)
        intensity_sup = np.max(self.im)

        intensities, phi_a, phi_b, phi = self.get_trheshold_isodata()

        theta = 55  # int((intensity_sup - intensity_inf) / 2)

        while (True):
            avg_1 = phi_a[theta]
            avg_2 = phi_b[theta + 1]
            new_theta = int((avg_1 + avg_2) / 2)
            if theta == new_theta:
                break
            theta = new_theta

        return theta


def gamma_correction(im, gamma=2):
    im = norm(im, a_new=0, b_new=1)
    return im ** gamma * 255


def binning(values, bins, ids=None):
    values = np.array(values)
    if ids is None:
        ids = np.arange(values.shape[0])
    inf_f = np.min(ids)
    sup_f = np.max(ids)

    new_ids = np.zeros(bins)
    new_values = np.zeros(bins)

    rs = np.linspace(0, 256, bins + 1)
    rs[bins] += 1
    for k in range(bins):
        start = rs[k]
        end = rs[k + 1]

        new_values[k] = np.sum(values[(ids >= start) & (ids < end)])
    rs[bins] -= 1

    return rs[:-1], new_values


class Histogram:
    def __init__(self, in_im):
        self.im = in_im
        self.size = 256

    def get_hist(self, bins=256):
        counts = np.zeros(self.size)
        intensities, pix_counts = np.unique(self.im, return_counts=True)

        for intensity, pix_count in zip(intensities, pix_counts):
            counts[intensity] = pix_count

        hist = counts / np.prod(self.im.shape)
        return binning(hist, bins, np.arange(self.size))

    def get_cum_hist(self, bins=256):
        ids, values = self.get_hist(bins)
        return ids, np.cumsum(values)

    def get_transform(self):
        return self.get_cum_hist()[1]

    def get_equalization_map(self):
        return np.rint(self.get_transform() * 255).astype(np.uint8)

    def get_inverse_transform(self):
        T = self.get_equalization_map()
        counts = np.zeros(self.size)

        for t in T:
            counts[t] += 1

        return np.cumsum(counts) / np.prod(T.shape)

    def get_inverse_equalization_map(self):
        return np.rint(self.get_inverse_transform() * 255).astype(np.uint8)

    #     def get_equalized_cum_hist(self):
    #         t = self.get_equalization_map()
    #         return t, self.get_cum_hist()[1][t]

    def equalize(self, target_im=None):
        if target_im is None:
            target_im = self.im
        #         target_im = cvu.norm(im).astype(np.uint8)
        return self.get_equalization_map()[target_im]


def hist_equalization(im):
    im_hist = np.zeros(256).astype(np.int32)
    a, cnts = np.unique(im, return_counts=True)
    for idx, cnt in zip(a, cnts):
        im_hist[idx] = cnt

    num_pixels = np.prod(im.shape)
    final_map = np.rint(np.cumsum(im_hist / num_pixels) * 255)

    squarer = lambda t: final_map[t]

    vfunc = np.vectorize(squarer)

    return vfunc(im)


#### Local-Filters


def gauss_deriv_filter(im, sigma):
    cval = 0
    sigma_range = 3
    x = np.arange(np.floor(-sigma_range * sigma + 0.5), np.floor(sigma_range * sigma + 0.5) + 1)
    gaussian = gauss(x, sigma)
    derivative = gaussdx(x, sigma)
    gaussian2d = gaussian.reshape(1, -1)
    derivative2d = derivative.reshape(1, -1)

    tmp = cv2.filter2D(im, -1, derivative2d)
    imDx = cv2.filter2D(tmp, -1, gaussian2d.T)

    tmp = cv2.filter2D(im, -1, gaussian2d)
    imDy = cv2.filter2D(tmp, -1, derivative2d.T)

    return imDx, imDy


def gauss_smooth_filter(im, sigma):
    sigma_range = 3
    ws = 2 * int(sigma_range * sigma) + 1
    x = np.arange(np.floor(-sigma_range * sigma + 0.5), np.floor(sigma_range * sigma + 0.5) + 1)
    gaussian = gauss(x, sigma).reshape(-1, 1)
    cval = 0
    tmp = cv2.filter2D(im, -1, gaussian)
    ii = cv2.filter2D(tmp, -1, gaussian.T)

    return ii


######################## IO

def concatenate(images, layout):
    nrows, ncols = layout
    nimages = len(images)
    nlayout = nrows * ncols
    if nimages != nlayout:
        raise IndexError(f"The layout supports {nlayout} but {nimages} images were provided.")
    rows = []
    for i in range(nrows):
        start = i * ncols
        end = start + ncols
        row = np.concatenate(images[start:end], axis=1)
        rows.append(row)

    final_im = np.concatenate(rows, axis=0)
    final_im = norm(final_im)

    return final_im.astype(np.uint8)


def window_stream():
    def load():
        def apply(images, layout):
            final = concatenate(images, layout)
            cv2.imshow("output", final)

        return apply

    class Container(object):
        def __init__(self):
            self.stream = load()

        def __enter__(self):
            return self.stream

        def __exit__(self, type, value, traceback):
            cv2.destroyAllWindows()

    return Container()


class MyCam(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def __enter__(self):
        return self.cap

    def __exit__(self, type, value, traceback):
        self.cap.release()


def cam_stream(wait_key=1, scale=1):
    def load():
        closed = False
        with MyCam() as cap:
            while (True):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                frame = scale_im(frame, scale)
                yield (ret, frame)
                if cv2.waitKey(wait_key) & 0xFF == ord('q'):
                    break

    class Container(object):
        def __init__(self):
            self.stream = load()

        def __enter__(self):
            return self.stream

        def __exit__(self, type, value, traceback):
            if type is not None:
                self.stream.throw(value)

    return Container()


def dir_stream(folder, wait_key=30, scale=1):
    def load():
        for filename in sorted(os.listdir(folder)):
            fullname = os.path.join(folder, filename)
            frame = cv2.imread(fullname)
            frame = scale_im(frame, scale)
            yield (), frame
            if cv2.waitKey(wait_key) & 0xFF == ord('q'):
                break

    class Container(object):
        def __init__(self):
            self.stream = load()

        def __enter__(self):
            return self.stream

        def __exit__(self, type, value, traceback):
            pass

    return Container()


def dir_images(path):
    return [im_name for im_name in glob.glob(path + "*.jpg")]
