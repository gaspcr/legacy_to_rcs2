__author__ = "furcelay"

from astropy.wcs import WCS
import numpy as np


deg2arcsec = 3600


def resample_image(img, original_pix_scale, lsst_pix_scale=0.2, drop_edge=5, out_size=None):
    """Resample an image to LSST pixel scale using reproject_adaptive from the reproject package.
    https://reproject.readthedocs.io/en/stable/

    :param img: 2D numpy array, input image to be resampled
    :param original_pix_scale: float, pixel scale of the input image in arcsec/pixel
    :param lsst_pix_scale: float, pixel scale of LSST in arcsec/pixel, default is 0.2 arcsec/pixel
    :param drop_edge: int, number of pixels to drop from each edge after resampling, default is 5.
        Requered to remove bad pixels introduced by the resampling process.
    :param out_size: tuple of int, desired output size in pixels (ny, nx).
        If None, size is determined by input image size and pixel scales.
    :return: 2D numpy array, resampled image to LSST pixel scale
    """
    # Imported lazily: reproject is only needed when an actual resample runs,
    # so the package stays importable (and the pipeline wiring testable)
    # without it installed.
    from reproject import reproject_adaptive

    img_wcs = WCS(naxis=2)
    img_wcs.wcs.cd = np.array([[-1, 0],
                               [ 0, 1]]) * original_pix_scale / deg2arcsec
    img_crpix = np.flip(np.array(img.shape)) / 2
    img_wcs.wcs.crpix = img_crpix

    img_field = np.array(img.shape) * original_pix_scale
    img_center = img_wcs.pixel_to_world(*((np.array(img.shape) - 1) / 2))

    if out_size is None:
        lsst_size = img_field // lsst_pix_scale
    else:
        lsst_size = out_size
    lsst_crpix = np.flip(lsst_size) / 2

    lsst_wcs = WCS(naxis=2)
    lsst_wcs.wcs.crpix = lsst_crpix
    lsst_wcs.wcs.cd = np.array([[-1, 0],
                                [ 0, 1]]) * lsst_pix_scale / deg2arcsec
    lsst_wcs.wcs.ctype = img_wcs.wcs.ctype

    img_scaled, _ = reproject_adaptive((img, img_wcs), lsst_wcs,
                                       shape_out=lsst_size.astype(int), conserve_flux=True,
                                       bad_fill_value=0)
    # drop edge that has bad pixels
    if drop_edge > 0:
        img_scaled = img_scaled[drop_edge:-drop_edge,
                                drop_edge:-drop_edge]
    return img_scaled