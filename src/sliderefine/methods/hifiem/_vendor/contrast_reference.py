"""HiFiEM contrast excerpt. See adjacent LICENSE and NOTICE before redistribution.

Computational body of adjust_histogram_v_3_02 from pinned upstream fltemd.py.
This is low-level author reference behavior (including input mutation and the
inclusive output-bin convention), NOT a safe default for AF or WSI inputs.
"""
import gc
import numpy as np
from scipy.ndimage import convolve1d, maximum_filter, minimum_filter
from scipy.stats import norm


class kernel:
    """Minimal compatibility shim for this function's explicit 1-D kernels."""
    def __init__(self, kern):
        self.kern = np.array(kern)

    def convn(self, input, mode="mirror", ndims=None):
        ndims = input.ndim if ndims is None else ndims
        output = input
        for ndim in range(ndims):
            output = convolve1d(output, self.kern, axis=-(ndim+1), mode=mode)
        return output


def adjust_histogram_v_3_02(image, drange = 255, dtype = np.uint8, bwmask = None, smoothing = 501,
                            method = "image", sigma = 7.5, offset = -15, ratio = 2, minmax_size = 9,
                            compress_range = None, compress_ratio = 0.1, clip = [-0.8, 0.8],
                            redistribute = "left", hp_sigma = 15, mean = None, save_memory = True):
    """Upstream contrast function; docstring shortened, computational body retained.

    Floating input outside bwmask may be mutated. Default output is uint8.
    Call the AF wrapper for validated normalized input and background preservation.
    """
    eps    = np.finfo(np.float32).eps
    drange = (0, 255) if drange is None else (0, drange) if np.isscalar(drange) else drange
    mean   = np.mean(drange) if mean is None else mean
    method = "image" if method is None else method.lower()
    dtype  = image.dtype if dtype is None else dtype
    clip = clip if (clip is None) or (not np.isscalar(clip)) else None if clip == 0 else (-abs(clip), abs(clip))
    redistribute = None if clip is None else redistribute if not (redistribute is None) else 'full'

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32)
    elif not bwmask is None:
        bwmask = bwmask & ~np.isnan(image)
    else:
        bwmask = ~np.isnan(image)

    if (bwmask is None) or np.all(bwmask):
        bwmask = None # we later multiply some results by bwmask
    else:
        image[~bwmask] = mean

    if (not ratio is None) and (ratio > 0):
        kernA = np.ones((smoothing,), dtype = np.float32) / np.float32(smoothing)
        kernA = kernel(np.correlate(kernA, kernA, mode = 'full'))
        if bwmask is None:
            lopass = kernA.convn(image, ndims = 2)
            hipass = image - lopass
            MULTI  = lambda vals, mean: ratio * (1 - norm.cdf(vals, loc = mean + offset, scale = sigma))
        else:
            bwmask = bwmask.astype(np.float32)
            hipass = bwmask * (image - kernA.convn(image * bwmask, ndims = 2) / (kernA.convn(bwmask, ndims = 2) + eps))
            lopass = image - hipass
            MULTI  = lambda vals, mean: ratio * bwmask * (1 - norm.cdf(vals, loc = mean + offset, scale = sigma))

        if method != "image":
            maxval = maximum_filter(image, size = minmax_size, mode = "mirror")
            minval = minimum_filter(image, size = minmax_size, mode = "mirror")
            middle = 0.5 * (maxval + minval)

        if compress_range is None:
            multi = MULTI(locals()[method], lopass)
            if method == "image":
                middle = 0.5 * (maximum_filter(image, size = minmax_size, mode = "mirror") + minimum_filter(image, size = minmax_size, mode = "mirror"))
        else:
            multi  = MULTI(locals()[method], np.maximum(compress_range[0], np.minimum(compress_range[1], lopass)))
            lopass = 0.5 * (1 - compress_ratio) * (compress_range[1] - compress_range[0]) + np.minimum(lopass, compress_range[0]) + np.maximum(0, lopass - compress_range[1]) \
                + compress_ratio * np.minimum(compress_range[1] - compress_range[0], np.maximum(0, lopass - compress_range[0]))
            image  = lopass + hipass
            middle = 0.5 * (maximum_filter(image, size = minmax_size, mode = "mirror") + minimum_filter(image, size = minmax_size, mode = "mirror"))

        if save_memory:
            if method != "image":
                del minval, maxval
            gc.collect()

        image  = image + multi * (image - middle)

    hipass = (image - mean) / hp_sigma
    ncdf = norm.cdf(hipass, 0, 1)
    # if save_memory: gc.collect()
    if not clip is None:
        cdf = norm.cdf(clip); # left and right normal cdf for clipped boundaries
        pdf = norm.pdf(clip); # left and right normal pdf for clipped boundaries

        delta   = (cdf[1] - cdf[0]) - 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])
        hipass  = np.clip(hipass - clip[0], 0, clip[1] - clip[0], out = hipass)
        hipass *= 0.5 * (2 * pdf[0] + (pdf[1] - pdf[0]) * hipass / (clip[1] - clip[0]))

        if redistribute.lower() == 'leftmost':
            mul  = delta / cdf[0] + 1
            ncdf = mul * np.minimum(ncdf, cdf[0]) + hipass + np.maximum(ncdf - cdf[1], 0)
        elif redistribute.lower() == 'left':
            mul  = delta / (cdf[0] + 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])) + 1
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + hipass) + np.maximum(ncdf - cdf[1], 0)
        else: # redistribute.lower() == 'full':
            mul  = 1 / (1 - delta)
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + hipass + np.maximum(ncdf - cdf[1], 0))

    # return np.clip(middle + (drange[1] - drange[0]) * (ncdf - 0.5), drange[0], drange[1]).astype(dtype)
    return np.clip(drange[0] + (drange[1] - drange[0] + 1) * ncdf, drange[0], drange[1]).astype(dtype, copy = False);
