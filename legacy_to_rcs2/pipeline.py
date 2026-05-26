"""Orchestration for the Legacy Survey DR10 -> RCS2 degradation pipeline.

Adapted from furcelay/HSC_to_LSST (F. Urcelay, 2024, MIT). The HSC-specific
input end (authenticated HSC cutout service + 2D PSF models) is replaced by
the Legacy Survey ends:
  - image cutouts via legacy_query.cutout.query_legacy_cutout;
  - per-band PSF FWHM and per-pixel RMS via legacy_query.tractor_props
    (no 2D PSF model is available, so a Gaussian PSF of the tractor FWHM
    is used by the degradation core).

The target survey is RCS2 (single-exposure CFHT/MegaCam frames), whose
observing conditions are drawn per band by an rcs2_sampler exposing
    rcs2_sampler[band].sample() -> {median, rms, seeing, exp_time,
                                    zero_point, gain}
(see legacy_to_rcs2.rcs2_props.rcs2_samplers).

Two ways to run:
  - live:  query_and_degrade / query_degrade_write (download + degrade);
  - cached: query_and_write (download once) then read_and_degrade /
    read_degrade_write (degrade the cached cutout, possibly many times
    with different sampled RCS2 conditions for augmentation).
"""

from contextlib import nullcontext
from multiprocessing import Lock
import warnings

import numpy as np
from astropy.io import fits

from legacy_to_rcs2.legacy_query.cutout import query_legacy_cutout
from legacy_to_rcs2.legacy_query.tractor_props import query_tractor_props
from legacy_to_rcs2.data_degradation.zero_point import zero_point_change
from legacy_to_rcs2.data_degradation.hsc_degradation import legacy_to_rcs2


# Legacy Survey DR10 coadds are calibrated in nanomaggies, i.e. AB zero
# point 22.5. This is the "original" zero point fed to zero_point_change.
LEGACY_ZERO_POINT = 22.5

# Native DECam pixel scale of Legacy DR10 coadds (arcsec/pixel).
LEGACY_PIX_SCALE = 0.262

# RCS2 / MegaCam output pixel scale (arcsec/pixel).
RCS2_PIX_SCALE = 0.185

# Final CNN stamp size: 65 px = 12 arcsec at the RCS2 pixel scale.
RCS2_STAMP_PIX = 65

# Default Legacy cutout download size (px, at LEGACY_PIX_SCALE). 80 px =
# ~21 arcsec, generous margin over the 12 arcsec stamp for PSF convolution
# and the flux-conserving resample edge.
LEGACY_CUTOUT_PIX = 80


def degrade_images(
    images,
    tractor_props,
    rcs2_sampler,
    bands='grz',
    legacy_pix_scale=LEGACY_PIX_SCALE,
    target_pix_scale=RCS2_PIX_SCALE,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    out_size_pix=RCS2_STAMP_PIX,
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Degrade Legacy DR10 images to RCS2-like images.

    :param images: list of 2D numpy arrays, one Legacy image per band
    :param tractor_props: dict {band: {'fwhm_arcsec': float, 'rms_pixel': float}}
        as returned by query_tractor_props; gives the Legacy PSF FWHM used
        to convolve up to the RCS2 seeing.
    :param rcs2_sampler: dict {band: BandSampler}; each .sample() returns the
        RCS2 conditions {median, rms, seeing, exp_time, zero_point, gain}
    :param bands: str, concatenated band letters (default 'grz')
    :param legacy_pix_scale: float, pixel scale of the input Legacy images
    :param target_pix_scale: float, RCS2/MegaCam output pixel scale
    :param zp_rms_frac_thresh: float, passed to zero_point_change; the
        target zero point is reached only if RCS2_rms * thresh > Legacy_rms,
        else an intermediate (colour-preserving) zero point is used
    :param zp_max_mag_change: float, max magnitude change allowed in the
        zero-point step; degradation fails if exceeded
    :param out_size_pix: int, output stamp side length in pixels
    :param log_file: str, path to log file
    :param log_lock: Lock to synchronize log writes
    :param log_prefix: str, prefix for log messages
    :return: (success: bool, degraded_images: list|None, mag_change: float)
    """
    # Sample RCS2 destination conditions, one coherent draw per band.
    rcs2_stats = [rcs2_sampler[band].sample() for band in bands]

    # MegaCam/Elixir PHOT_C is a per-second (ADU/s) zero point, so after
    # zero_point_change the image is a count RATE. The sampled sky rms is in
    # total ADU (measured on the raw frame), so divide by exp_time to put the
    # noise in the same ADU/s units as the image and the zero point.
    rcs2_rms_cps = [s['rms'] / s['exp_time'] for s in rcs2_stats]
    rcs2_zero_points = [s['zero_point'] for s in rcs2_stats]
    images, mag_change = zero_point_change(
        images,
        LEGACY_ZERO_POINT,
        rcs2_zero_points,
        rcs2_rms_cps,
        rms_frac_thresh=zp_rms_frac_thresh,
    )
    if mag_change > zp_max_mag_change:
        log_message(
            log_file, log_lock,
            f"Error scaling zero point: magnitude change {mag_change} "
            f"exceeds maximum {zp_max_mag_change}",
            log_prefix,
        )
        return False, None, 0.0

    degraded_images = []
    for b, band in enumerate(bands):
        stats = rcs2_stats[b]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                deg_img_b = legacy_to_rcs2(
                    images[b],
                    band,
                    exp_time=stats['exp_time'],
                    lsst_zero_point=stats['zero_point'],
                    # RCS2 frames are single exposures: no coadd->single
                    # exposure zero-point correction is needed.
                    lsst_single_exposure_zero_point=None,
                    lsst_ccd_gain=stats['gain'],
                    # Legacy PSF as a Gaussian of the tractor FWHM (no 2D PSF).
                    hsc_fwhm=tractor_props[band]['fwhm_arcsec'],
                    hsc_psf=None,
                    hsc_pix_scale=legacy_pix_scale,
                    target_pix_scale=target_pix_scale,
                    lsst_fwhm=stats['seeing'],
                    # ADU/s to match the count-rate image (see rcs2_rms_cps above).
                    background_noise=stats['rms'] / stats['exp_time'],
                    background_median=stats['median'],
                    psf_transform=True,
                    add_poisson_noise=True,
                    add_background_noise=True,
                    use_noise_diff=True,
                    out_size=out_size_pix,
                )
            # Keep going regardless of the error (common: ValueError when the
            # Legacy noise already exceeds the RCS2 target, see add_noise).
            except Exception as e:
                log_message(
                    log_file, log_lock,
                    f"Error processing band {band}: {type(e)} {e}",
                    log_prefix,
                )
                return False, None, 0.0
        degraded_images.append(deg_img_b)
    return True, degraded_images, mag_change


def query_and_degrade(
    ra,
    dec,
    rcs2_sampler,
    semaphore=None,
    bands='grz',
    layer='ls-dr10',
    legacy_pix_scale=LEGACY_PIX_SCALE,
    target_pix_scale=RCS2_PIX_SCALE,
    cutout_size_pix=LEGACY_CUTOUT_PIX,
    tractor_search_radius_arcsec=60.0,
    query_timeout=60,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    out_size_pix=RCS2_STAMP_PIX,
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Query Legacy DR10 data at (ra, dec) and degrade it to RCS2-like images.

    :param ra: float, right ascension in degrees (J2000)
    :param dec: float, declination in degrees (J2000)
    :param rcs2_sampler: dict {band: BandSampler}
    :param semaphore: optional multiprocessing.BoundedSemaphore to throttle
        concurrent HTTP queries; if None, no throttling
    :param bands: str, concatenated band letters
    :param layer: str, Legacy Survey layer (default 'ls-dr10')
    :param legacy_pix_scale: float, Legacy cutout pixel scale (native 0.262)
    :param target_pix_scale: float, RCS2 output pixel scale
    :param cutout_size_pix: int, Legacy cutout download size in pixels
    :param tractor_search_radius_arcsec: float, cone radius for the nearest
        tractor object used to read the PSF/noise properties
    :param query_timeout: float, per-request HTTP timeout in seconds
    :param zp_rms_frac_thresh: float, see degrade_images
    :param zp_max_mag_change: float, see degrade_images
    :param out_size_pix: int, output stamp side length in pixels
    :param log_file: str, path to log file
    :param log_lock: Lock to synchronize log writes
    :param log_prefix: str, prefix for log messages
    :return: (success: bool, degraded_images: list|None, mag_change: float)
    """
    conn_guard = semaphore if semaphore is not None else nullcontext()
    try:
        with conn_guard:
            cutout = query_legacy_cutout(
                ra, dec, bands=bands, layer=layer,
                pixscale=legacy_pix_scale, size_pix=cutout_size_pix,
                timeout=query_timeout,
            )
            tractor_props = query_tractor_props(
                ra, dec, bands=bands,
                search_radius_arcsec=tractor_search_radius_arcsec,
                pixscale=legacy_pix_scale, timeout=query_timeout,
            )
    # Keep going regardless of the error (network / footprint / parse issues).
    except Exception as e:
        log_message(
            log_file, log_lock,
            f"Error querying Legacy data: {type(e)} {e}",
            log_prefix,
        )
        return False, None, 0.0

    images = [cutout[band]['image'] for band in bands]
    return degrade_images(
        images,
        tractor_props,
        rcs2_sampler,
        bands=bands,
        legacy_pix_scale=legacy_pix_scale,
        target_pix_scale=target_pix_scale,
        zp_rms_frac_thresh=zp_rms_frac_thresh,
        zp_max_mag_change=zp_max_mag_change,
        out_size_pix=out_size_pix,
        log_file=log_file,
        log_lock=log_lock,
        log_prefix=log_prefix,
    )


def query_degrade_write(
    out_filename,
    ra,
    dec,
    rcs2_sampler,
    semaphore=None,
    bands='grz',
    layer='ls-dr10',
    legacy_pix_scale=LEGACY_PIX_SCALE,
    target_pix_scale=RCS2_PIX_SCALE,
    cutout_size_pix=LEGACY_CUTOUT_PIX,
    tractor_search_radius_arcsec=60.0,
    query_timeout=60,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    out_size_pix=RCS2_STAMP_PIX,
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Query Legacy DR10, degrade to RCS2-like images, and write them.

    :param out_filename: str, output path without band suffix; each band is
        written to ``{out_filename}_{band}.fits``
    :return: (success: bool, mag_change: float)
    (remaining parameters: see query_and_degrade)
    """
    success, degraded_images, mag_change = query_and_degrade(
        ra, dec, rcs2_sampler, semaphore,
        bands=bands, layer=layer,
        legacy_pix_scale=legacy_pix_scale, target_pix_scale=target_pix_scale,
        cutout_size_pix=cutout_size_pix,
        tractor_search_radius_arcsec=tractor_search_radius_arcsec,
        query_timeout=query_timeout,
        zp_rms_frac_thresh=zp_rms_frac_thresh,
        zp_max_mag_change=zp_max_mag_change,
        out_size_pix=out_size_pix,
        log_file=log_file, log_lock=log_lock, log_prefix=log_prefix,
    )
    if success:
        write_degraded_images(out_filename, degraded_images, bands)
    return success, mag_change


def query_and_write(
    out_filename,
    ra,
    dec,
    semaphore=None,
    bands='grz',
    layer='ls-dr10',
    legacy_pix_scale=LEGACY_PIX_SCALE,
    cutout_size_pix=LEGACY_CUTOUT_PIX,
    tractor_search_radius_arcsec=60.0,
    query_timeout=60,
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Download a Legacy DR10 cutout and cache it (without degrading).

    The cached cutout carries the tractor PSF FWHM and per-pixel RMS in its
    FITS header so that read_and_degrade can run offline. Useful to download
    a source once and then degrade it many times with different sampled RCS2
    conditions (data augmentation).

    :param out_filename: str, output path without band suffix; each band is
        cached to ``{out_filename}_img_{band}.fits``
    :return: success (bool)
    (remaining parameters: see query_and_degrade)
    """
    conn_guard = semaphore if semaphore is not None else nullcontext()
    try:
        with conn_guard:
            cutout = query_legacy_cutout(
                ra, dec, bands=bands, layer=layer,
                pixscale=legacy_pix_scale, size_pix=cutout_size_pix,
                timeout=query_timeout,
            )
            tractor_props = query_tractor_props(
                ra, dec, bands=bands,
                search_radius_arcsec=tractor_search_radius_arcsec,
                pixscale=legacy_pix_scale, timeout=query_timeout,
            )
    except Exception as e:
        log_message(
            log_file, log_lock,
            f"Error querying Legacy data: {type(e)} {e}",
            log_prefix,
        )
        return False

    write_cutout_cache(out_filename, cutout, tractor_props,
                       legacy_pix_scale, bands)
    return True


def read_and_degrade(
    in_filename,
    rcs2_sampler,
    bands='grz',
    target_pix_scale=RCS2_PIX_SCALE,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    out_size_pix=RCS2_STAMP_PIX,
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Read a cached Legacy cutout and degrade it to RCS2-like images.

    :param in_filename: str, input path without band suffix (as written by
        query_and_write)
    :return: (success: bool, degraded_images: list|None, mag_change: float)
    (remaining parameters: see degrade_images)
    """
    images, tractor_props, legacy_pix_scale = read_cutout_cache(in_filename, bands)
    return degrade_images(
        images,
        tractor_props,
        rcs2_sampler,
        bands=bands,
        legacy_pix_scale=legacy_pix_scale,
        target_pix_scale=target_pix_scale,
        zp_rms_frac_thresh=zp_rms_frac_thresh,
        zp_max_mag_change=zp_max_mag_change,
        out_size_pix=out_size_pix,
        log_file=log_file,
        log_lock=log_lock,
        log_prefix=log_prefix,
    )


def read_degrade_write(
    in_filename,
    out_filename,
    rcs2_sampler,
    bands='grz',
    target_pix_scale=RCS2_PIX_SCALE,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    out_size_pix=RCS2_STAMP_PIX,
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Read a cached Legacy cutout, degrade it, and write the RCS2-like images.

    :param in_filename: str, cached cutout path without band suffix
    :param out_filename: str, output path without band suffix
    :return: (success: bool, mag_change: float)
    (remaining parameters: see degrade_images)
    """
    success, degraded_images, mag_change = read_and_degrade(
        in_filename, rcs2_sampler,
        bands=bands, target_pix_scale=target_pix_scale,
        zp_rms_frac_thresh=zp_rms_frac_thresh,
        zp_max_mag_change=zp_max_mag_change,
        out_size_pix=out_size_pix,
        log_file=log_file, log_lock=log_lock, log_prefix=log_prefix,
    )
    if success:
        write_degraded_images(out_filename, degraded_images, bands)
    return success, mag_change


def write_cutout_cache(out_filename, cutout, tractor_props,
                       legacy_pix_scale=LEGACY_PIX_SCALE, bands='grz'):
    """Cache a Legacy cutout per band, with tractor props in the header.

    The PSF FWHM, per-pixel RMS and pixel scale are stored as the header
    cards LEGFWHM, LEGRMS and LEGPIXSC so the cutout is self-describing.

    :param out_filename: str, output path without band suffix
    :param cutout: dict {band: {'image': 2D ndarray, 'hdr': Header}} from
        query_legacy_cutout
    :param tractor_props: dict {band: {'fwhm_arcsec', 'rms_pixel', ...}}
    :param legacy_pix_scale: float, pixel scale to record in the header
    :param bands: str, concatenated band letters
    :return: None
    """
    for band in bands:
        hdr = cutout[band]['hdr'].copy()
        hdr['LEGFWHM'] = (tractor_props[band]['fwhm_arcsec'],
                          'Legacy PSF FWHM [arcsec] (ls_dr10.tractor)')
        hdr['LEGRMS'] = (tractor_props[band]['rms_pixel'],
                         'Legacy per-pixel RMS [nanomaggies]')
        hdr['LEGPIXSC'] = (legacy_pix_scale, 'Legacy pixel scale [arcsec/pix]')
        hdu = fits.PrimaryHDU(data=cutout[band]['image'], header=hdr)
        hdu.writeto(f"{out_filename}_img_{band}.fits", overwrite=True)


def read_cutout_cache(in_filename, bands='grz'):
    """Read a cached Legacy cutout written by write_cutout_cache.

    :param in_filename: str, input path without band suffix
    :param bands: str, concatenated band letters
    :return: (images: list of 2D ndarray, tractor_props: dict, legacy_pix_scale: float)
    """
    images = []
    tractor_props = {}
    legacy_pix_scale = LEGACY_PIX_SCALE
    for band in bands:
        image, hdr = fits.getdata(f"{in_filename}_img_{band}.fits", header=True)
        images.append(np.asarray(image, dtype=np.float32))
        tractor_props[band] = {
            'fwhm_arcsec': float(hdr['LEGFWHM']),
            'rms_pixel': float(hdr['LEGRMS']),
        }
        if 'LEGPIXSC' in hdr:
            legacy_pix_scale = float(hdr['LEGPIXSC'])
    return images, tractor_props, legacy_pix_scale


def write_degraded_images(filename, images, bands='grz'):
    """Write degraded RCS2-like images to .fits files, one per band.

    :param filename: str, output path without band suffix
    :param images: list of 2D ndarrays, one per band
    :param bands: str, concatenated band letters
    :return: None
    """
    for b, band in enumerate(bands):
        hdu = fits.PrimaryHDU(data=images[b])
        hdu.writeto(f"{filename}_{band}.fits", overwrite=True)


def log_message(log_file, log_lock, message, log_prefix=''):
    """Append a message to a log file with thread/process safety.

    :param log_file: str, path to log file
    :param log_lock: Lock to synchronize log writes
    :param message: str, message to log
    :param log_prefix: str, prefix for the message
    :return: None
    """
    with log_lock:
        with open(log_file, 'a') as f:
            f.write(log_prefix + message + '\n')
