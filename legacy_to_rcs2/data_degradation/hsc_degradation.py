__author__ = "furcelay"

from astropy.nddata import Cutout2D
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from legacy_to_rcs2.data_degradation.psf import degrade_psf
from legacy_to_rcs2.data_degradation.resample import resample_image
from legacy_to_rcs2.data_degradation.noise import add_noise


def legacy_to_rcs2(
        hsc_img,
        lsst_band,
        exp_time=30.0,
        lsst_zero_point=27.0,
        lsst_single_exposure_zero_point=None,
        lsst_ccd_gain=0.7,
        hsc_fwhm=0.6,
        hsc_psf=None,
        hsc_pix_scale=0.168,
        lsst_fwhm=None,
        lsst_psf=None,
        background_noise=None,
        background_median=0,
        psf_transform=True,
        add_poisson_noise=True,
        add_background_noise=True,
        use_noise_diff=True,
        out_size=64
):
    """Degrade HSC image to LSST-like image.
    :param hsc_img: 2D numpy array of HSC image
    :param lsst_band: str, LSST band ('u','g','r','i','z','y')
    :param exp_time: float, LSST exposure time in seconds
    :param lsst_zero_point: float, LSST zero point magnitude of the coadd or single exposure
    :param lsst_single_exposure_zero_point: float, LSST zero point magnitude of a single exposure
    :param lsst_ccd_gain: float, LSST CCD gain in e-/ADU
    :param hsc_fwhm: float, FWHM of HSC PSF in arcsec, only used if the PSF model is not provided
    :param hsc_psf: 2D numpy array of PSF model for HSC image, if None a Gaussian PSF with hsc_fwhm is used
    :param hsc_pix_scale: float, pixel scale of HSC image in arcsec/pixel
    :param lsst_fwhm: float, FWHM of LSST PSF in arcsec, only used if the PSF model is not provided
    :param lsst_psf: 2D numpy array of PSF model for LSST image, if None a Gaussian PSF with lsst_fwhm is used
    :param background_noise: float, background noise RMS for LSST image
    :param background_median: float, background median level for LSST image
    :param psf_transform: bool, whether to perform PSF transformation from HSC to LSST
    :param add_poisson_noise: bool, whether to add Poisson noise
    :param add_background_noise: bool, whether to add background noise
    :param use_noise_diff: bool, whether to consider the HSC noise when adding LSST background noise
    :param out_size: int, output image size in pixels (output image is square)
    :return: 2D numpy array of degraded LSST-like image
    """
    lsst_band_props = LSST(band=lsst_band).kwargs_single_band()
    if lsst_fwhm is None:
        lsst_fwhm = lsst_band_props['seeing']
    lsst_pix_scale = lsst_band_props['pixel_scale']
    # PSF
    if psf_transform:
        hsc_img_conv = degrade_psf(
                hsc_img,
                original_fwhm=hsc_fwhm,
                target_fwhm=lsst_fwhm,
                original_psf=hsc_psf,
                target_psf=lsst_psf,
                original_pix_scale=hsc_pix_scale,
                target_pix_scale=lsst_pix_scale,
                max_iters=3,
                thresh=0.01
            )
    else:
        hsc_img_conv = hsc_img

    # resampling
    hsc_img_conv_scaled = resample_image(hsc_img_conv, hsc_pix_scale, lsst_pix_scale, drop_edge=0)
    # cut to out size, centered
    cy = (hsc_img_conv_scaled.shape[0] - 1) / 2
    cx = (hsc_img_conv_scaled.shape[1] - 1) / 2
    hsc_img_conv_scaled = Cutout2D(hsc_img_conv_scaled, (cy, cx), (out_size, out_size)).data
    # noise
    if add_poisson_noise or add_background_noise:
        hsc_img_conv_scaled_noise = add_noise(hsc_img_conv_scaled,
                                              lsst_band_props,
                                              exp_time,
                                              lsst_zero_point,
                                              background_noise=background_noise,
                                              background_median=background_median,
                                              add_poisson_noise=add_poisson_noise,
                                              add_background_noise=add_background_noise,
                                              use_noise_diff=use_noise_diff,
                                              single_exposure_zero_point=lsst_single_exposure_zero_point,
                                              ccd_gain=lsst_ccd_gain
                                              )
    else:
        hsc_img_conv_scaled_noise = hsc_img_conv_scaled

    return hsc_img_conv_scaled_noise
