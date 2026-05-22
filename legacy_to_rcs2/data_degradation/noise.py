__author__ = "furcelay"

from legacy_to_rcs2.utils import photutils_background_iterative
from lenstronomy.Util import data_util
import numpy as np


def add_noise(img,
              lsst_band_props,
              exp_time=30.0,
              zero_point=27.0,
              background_noise=None,
              background_median=0,
              add_poisson_noise=True,
              add_background_noise=True,
              use_noise_diff=True,
              single_exposure_zero_point=None,
              ccd_gain=0.7,
              ):
    """Add noise to an image to match LSST-like observations.

    :param img: 2D numpy array of input image already scaled to LSST zero point
    :param lsst_band_props: list of LSST band properties (see lenstronomy.SimulationAPI.ObservationConfig.LSST.LSST)
    :param exp_time: float, LSST exposure time in seconds
    :param zero_point: float, LSST zero point magnitude of the coadd or single exposure
    :param background_noise: float, background noise RMS for LSST image
    :param background_median: float, background median level for LSST image
    :param add_poisson_noise: bool, whether to add Poisson noise
    :param add_background_noise: bool, whether to add background noise
    :param use_noise_diff: bool, whether to consider the original noise when adding LSST background noise
    :param single_exposure_zero_point: float, LSST zero point magnitude of a single exposure, used to scale to e- counts
    :param ccd_gain: float, LSST CCD gain in e-/ADU
    :return: 2D numpy array of image with added noise
    """

    img_median, img_std, _ = photutils_background_iterative(img)
    img = img - img_median

    num_exposures = exp_time / 15

    if single_exposure_zero_point is not None:
        # Correct for the difference in zero points from the single exposure.
        # Single exposure is in e-/s and produce the correct Poisson noise
        zero_point_scale = 10**((zero_point - single_exposure_zero_point) / 2.5)
    else:
        zero_point_scale = 1

    # image with poisson noise in e-/s
    if add_poisson_noise:
        img_positive = np.where(img > 0, img, 0)
        cps_to_electrons = exp_time * ccd_gain / zero_point_scale
        img = np.random.poisson(lam=img_positive * cps_to_electrons) / cps_to_electrons

    if add_background_noise:
        if background_noise is None:
            sky_brightness_cps = data_util.magnitude2cps(
                    lsst_band_props['sky_brightness'],
                    magnitude_zero_point=zero_point,
                )

            bkg_noise = data_util.bkg_noise(
                    lsst_band_props['read_noise'],
                    15,
                    sky_brightness_cps,
                    lsst_band_props['pixel_scale'],
                    num_exposures,
                )
        else:
            bkg_noise = background_noise
        if bkg_noise > img_std:
            if use_noise_diff:
                noise = np.random.normal(scale=np.sqrt(bkg_noise**2 - img_std**2),
                                         size=img.shape)
                img = img + noise
            else:
                noise = np.random.normal(scale=bkg_noise, size=img.shape)
                img = img + noise
        else:
            # warnings.warn("Warning: original noise is larger than target")
            raise ValueError("Background noise is larger than target")

    img_median = photutils_background_iterative(img)[0]
    img += background_median - img_median

    return img
