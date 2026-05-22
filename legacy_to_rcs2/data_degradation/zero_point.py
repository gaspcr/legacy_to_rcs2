__author__ = "furcelay"

from legacy_to_rcs2.utils import photutils_background_iterative
import numpy as np


def zero_point_change(images, original_zero_points, lsst_zero_points, lsst_rms, rms_frac_thresh=0.1):
    """
    Changes the zero point of an image from original to target.
    If the noise allows it, it reaches the target zero point,
    in other case it reaches an intermediate point preserving
    the colors.

    :param images: list of 2D np.arrays for each band
    :param original_zero_points: list of original zero points for each band (27 for HSC)
    :param lsst_zero_points: list of target zero points for each band (27 for coadded LSST)
    :param lsst_rms: list of target rms values for each band
    :param rms_frac_thresh: fraction of target rms to reach on the scaled HSC image.
        If LSST_rms * rms_frac_thresh > original_rms, the target zero point is reached.
        otherwise an intermediate zero point is used perserving colors.
    :return: list of scaled images, magnitude change applied
    """
    original_zero_points = np.asarray(original_zero_points)
    lsst_zero_points = np.asarray(lsst_zero_points)
    lsst_rms = np.asarray(lsst_rms)
    original_rms = np.zeros(len(images))
    for i, img in enumerate(images):
        original_rms[i] = photutils_background_iterative(img)[1]
    max_zp_diff = 2.5 * np.log10(lsst_rms * rms_frac_thresh / original_rms)
    zp_diff = lsst_zero_points - original_zero_points
    if (zp_diff < max_zp_diff).all():
        # set to target zp
        scale = 10**(zp_diff / 2.5)
        mag_change = 0
    else:
        # scale to intermediate value that perserves colors:
        band_ref = np.argmin(max_zp_diff)
        original_zp_ref = original_zero_points[band_ref] if original_zero_points.size > 1 else original_zero_points
        target_zp_ref = max_zp_diff[band_ref] + original_zp_ref
        mag_change = lsst_zero_points[band_ref] - target_zp_ref
        target_zp = lsst_zero_points - mag_change
        zp_diff = target_zp - original_zero_points
        scale = 10**(zp_diff / 2.5)
    return [img * s for img, s in zip(images, scale)], mag_change
