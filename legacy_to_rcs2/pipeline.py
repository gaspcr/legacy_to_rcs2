__author__ = "furcelay"

from legacy_to_rcs2.hsc_query import query_hsc, query_hsc_binary
from legacy_to_rcs2.data_degradation.zero_point import zero_point_change
from legacy_to_rcs2.data_degradation.hsc_degradation import legacy_to_rcs2
from multiprocessing import Lock
from astropy.wcs import WCS
from astropy.io import fits
import warnings
import os.path


# used in case the PSF is not available
HSC_MEAN_FWHM = {
    'g': 0.71,
    'r': 0.74,
    'i': 0.53,
    'z': 0.56,
    'y': 0.53
}

# LSST single exposure average zero points
LSST_SINGLE_EXP_AVG_ZERO_POINTS = {
    'g': 32.33,
    'r': 32.17,
    'i': 31.85,
    'z': 31.45,
    'y': 30.63
}


def degrade_images(
    images,
    psfs,
    dp0_sampler,
    hsc_pix_scale=0.168,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    lsst_size_pix=41,
    bands='grizy',
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Degrade HSC images to LSST-like images.

    :param images: list of HSC images to degrade
    :param psfs: list of HSC PSF images corresponding to the HSC images
    :param dp0_sampler: dict(BandSampler) sampler object to sample LSST-like conditions of each band
    :param hsc_pix_scale: float pixel scale of HSC images in arcsec/pix
    :param zp_rms_frac_thresh: when scaling the zero point, the magnitude change must satisfy
        LSST_rms * zp_rms_frac_thresh > HSC_rms
    :param zp_max_mag_change: float maximum magnitude change allowed when changing zero point.
        If exceeded, the degradation fails.
    :param lsst_size_pix: int size of the output LSST-like images in pixels (square)
    :param bands: list of str bands corresponding to the images
    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param log_prefix: str prefix to add to log messages
    :return: success (bool), list of degraded images (or None if failed), magnitude change applied (float)
    """
    dp0_stats = []
    for band in bands:
        dp0_stats.append(dp0_sampler[band].sample())

    # change zero points
    dp0_rms = [dp0_stats[b]['rms'] for b in range(5)]
    dp0_zero_points = [dp0_stats[b]['zero_point'] for b in range(5)]
    images, mag_change = zero_point_change(images,
                                           27,
                                           dp0_zero_points,
                                           dp0_rms,
                                           rms_frac_thresh=zp_rms_frac_thresh)
    if mag_change > zp_max_mag_change:
        log_message(log_file, log_lock,
                    f"Error scaling zero point: magnitude change {mag_change} exceeds maximum {zp_max_mag_change}",
                    log_prefix)
        return False, None, 0.0

    degraded_images = []
    for b, band in enumerate(bands):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                deg_img_b = legacy_to_rcs2(images[b],
                                        band,
                                        exp_time=dp0_stats[b]['exp_time'],
                                        lsst_zero_point=dp0_stats[b]['zero_point'],
                                        lsst_single_exposure_zero_point=LSST_SINGLE_EXP_AVG_ZERO_POINTS[band],
                                        lsst_ccd_gain=0.7,
                                        hsc_fwhm=HSC_MEAN_FWHM[band],
                                        hsc_psf=psfs[b],
                                        hsc_pix_scale=hsc_pix_scale,
                                        lsst_fwhm=dp0_stats[b]['seeing'],
                                        background_noise=dp0_stats[b]['rms'],
                                        background_median=dp0_stats[b]['median'],
                                        psf_transform=True,
                                        add_poisson_noise=True,
                                        add_background_noise=True,
                                        use_noise_diff=True,
                                        out_size=lsst_size_pix
                                        )
            # this needs to keep going regardless of the error (common: ValueError)
            except Exception as e:
                log_message(log_file, log_lock,
                            f"Error processing band {band}: {type(e)} {e}",
                            log_prefix)
                return False, None, 0.0
        degraded_images.append(deg_img_b)
    return True, degraded_images, mag_change


def query_and_degrade(
    ra,
    dec,
    dp0_sampler,
    semaphore,
    username=None,
    password=None,
    query_timeuot=60,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    hsc_size_arcsec=20,
    lsst_size_pix=41,
    field="pdr3_wide",
    bands='grizy',
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Query HSC data and degrade it to LSST-like images.
    :param ra: float right ascension of the target in degrees
    :param dec: float declination of the target in degrees
    :param dp0_sampler: dict(BandSampler) sampler object to sample LSST-like conditions of each band
    :param semaphore: multiprocessing.BoundedSemaphore object to limit concurrent queries
    :param username: str HSC username for authentication
    :param password: str HSC password for authentication
    :param query_timeuot: int timeout for the HSC query in seconds
    :param zp_rms_frac_thresh: when scaling the zero point, the magnitude change must satisfy
        LSST_rms * zp_rms_frac_thresh > HSC_rms
    :param zp_max_mag_change: float maximum magnitude change allowed when changing zero point.
        If exceeded, the degradation fails.
    :param hsc_size_arcsec: int size of the HSC cutout to query in arcseconds
    :param lsst_size_pix: int size of the output LSST-like images in pixels (square)
    :param field: str HSC data release to query
    :param bands: list of str bands to query and degrade
    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param log_prefix: str prefix to add to log messages
    :return: success (bool), list of degraded images (or None if failed), magnitude change applied (float)
    """
    try:
        hsc_data = query_hsc(ra, dec, semaphore, username, password, query_timeuot, hsc_size_arcsec, field)
    # this needs to keep going regardless of the error (common: OSError and tarfile.ReadError)
    except Exception as e:
        log_message(log_file, log_lock,
                    f"Error querying HSC data: {type(e)} {e}",
                    log_prefix)
        return False, None, 0.0
    for band in bands:
        if not hsc_data[band]:
            log_message(log_file, log_lock,
                        f"Error querying HSC data: missing band {band}",
                        log_prefix)
            return False, None, 0.0

    pix_scale = WCS(hsc_data[bands[0]]['hdr']).wcs.cd[1, 1] * 3600

    images = [hsc_data[band]['image'] for band in bands]
    psfs = [hsc_data[band].get('psf', None) for band in bands]
    return degrade_images(
        images,
        psfs,
        dp0_sampler,
        pix_scale,
        zp_rms_frac_thresh,
        zp_max_mag_change,
        lsst_size_pix,
        bands,
        log_file,
        log_lock,
        log_prefix,
    )


def query_degrade_write(
        out_filename,
        ra,
        dec,
        dp0_sampler,
        semaphore,
        username=None,
        password=None,
        query_timeuot=60,
        zp_rms_frac_thresh=0.3,
        zp_max_mag_change=2,
        hsc_size_arcsec=20,
        lsst_size_pix=41,
        field="pdr3_wide",
        bands='grizy',
        log_file="log.txt",
        log_lock=Lock(),
        log_prefix='',
):
    """
    Query HSC data, degrade it to LSST-like images, and save it to a file

    :param out_filename: str path to output file (without band suffix)
    :param ra: float right ascension of the target in degrees
    :param dec: float declination of the target in degrees
    :param dp0_sampler: dict(BandSampler) sampler object to sample LSST-like conditions of each band
    :param semaphore: multiprocessing.BoundedSemaphore object to limit concurrent queries
    :param username: str HSC username for authentication
    :param password: str HSC password for authentication
    :param query_timeuot: int timeout for the HSC query in seconds
    :param zp_rms_frac_thresh: when scaling the zero point, the magnitude change must
        satisfy LSST_rms * zp_rms_frac_thresh > HSC_rms
    :param zp_max_mag_change: float maximum magnitude change allowed when changing zero point.
        If exceeded, the degradation fails.
    :param hsc_size_arcsec: int size of the HSC cutout to query in arcseconds
    :param lsst_size_pix: int size of the output LSST-like images in pixels (square)
    :param field: str HSC data release to query
    :param bands: list of str bands to query and degrade
    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param log_prefix: str prefix to add to log messages
    :return: success (bool), magnitude change applied (float)
    """
    success, degraded_images, mag_change = query_and_degrade(
        ra,
        dec,
        dp0_sampler,
        semaphore,
        username,
        password,
        query_timeuot,
        zp_rms_frac_thresh,
        zp_max_mag_change,
        hsc_size_arcsec,
        lsst_size_pix,
        field,
        bands,
        log_file,
        log_lock,
        log_prefix,
    )
    if success:
        write_degraded_images(
            out_filename,
            degraded_images,
            bands
        )
    return success, mag_change


def query_and_write(
    out_filename,
    ra,
    dec,
    semaphore,
    username=None,
    password=None,
    query_timeuot=60,
    hsc_size_arcsec=20,
    field="pdr3_wide",
    bands='grizy',
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
    require_psf=True,
):
    """
    Query HSC data and save it to a file (without degrading it)

    :param out_filename: str path to output file (without band suffix)
    :param ra: float right ascension of the target in degrees
    :param dec: float declination of the target in degrees
    :param semaphore: multiprocessing.BoundedSemaphore object to limit concurrent queries
    :param username: str HSC username for authentication
    :param password: str HSC password for authentication
    :param query_timeuot: int timeout for the HSC query in seconds
    :param hsc_size_arcsec: int size of the HSC cutout to query in arcseconds
    :param field: str HSC data release to query
    :param bands: list of str bands to query
    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param log_prefix: str prefix to add to log messages
    :param require_psf: bool whether to require PSF data for each band
    :return: success (bool)
    """
    try:
        hsc_data = query_hsc_binary(ra, dec, semaphore, username, password, query_timeuot, hsc_size_arcsec, field)
    # this needs to keep going regardless of the error (common: OSError and tarfile.ReadError)
    except Exception as e:
        log_message(log_file, log_lock,
                    f"Error querying HSC data: {type(e)} {e}",
                    log_prefix)
        return False
    for band in bands:
        if not hsc_data[band]:
            log_message(log_file, log_lock,
                        f"Error querying HSC data: missing band {band}",
                        log_prefix)
            return False
        elif require_psf:
            if 'psf' not in hsc_data[band]:
                log_message(log_file, log_lock,
                            f"Error querying HSC data: missing PSF for band {band}",
                            log_prefix)
                return False
    write_binary_images(
        f"{out_filename}_img",
        [hsc_data[band]['image'] for band in bands],
        bands
    )
    write_binary_images(
        f"{out_filename}_psf",
        [hsc_data[band]['psf'] for band in bands],
        bands
    )
    return True


def read_and_degrade(
    in_filename,
    dp0_sampler,
    zp_rms_frac_thresh=0.3,
    zp_max_mag_change=2,
    lsst_size_pix=41,
    bands='grizy',
    log_file="log.txt",
    log_lock=Lock(),
    log_prefix='',
):
    """Read HSC data from file and degrade it to LSST-like images.

    :param in_filename: str path to input file (without band suffix)
    :param dp0_sampler: dict(BandSampler) sampler object to sample LSST-like conditions of each band
    :param zp_rms_frac_thresh: when scaling the zero point, the magnitude change must satisfy
        LSST_rms * zp_rms_frac_thresh > HSC_rms
    :param zp_max_mag_change: float maximum magnitude change allowed when changing zero point.
        If exceeded, the degradation fails.
    :param lsst_size_pix: int size of the output LSST-like images in pixels (square)
    :param bands: list of str bands corresponding to the images
    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param log_prefix: str prefix to add to log messages
    :return: success (bool), list of degraded images (or None if failed), magnitude change applied (float)
    """
    images, psfs, pix_scale = read_images(in_filename, bands)
    return degrade_images(
        images,
        psfs,
        dp0_sampler,
        pix_scale,
        zp_rms_frac_thresh,
        zp_max_mag_change,
        lsst_size_pix,
        bands,
        log_file,
        log_lock,
        log_prefix,
    )


def read_degrade_write(
        in_filename,
        out_filename,
        dp0_sampler,
        zp_rms_frac_thresh=0.3,
        zp_max_mag_change=2,
        lsst_size_pix=41,
        bands='grizy',
        log_file="log.txt",
        log_lock=Lock(),
        log_prefix='',
):
    """
    Read HSC data from file, degrade it to LSST-like images, and save it to a file

    :param in_filename: str path to input file (without band suffix)
    :param out_filename: str path to output file (without band suffix)
    :param dp0_sampler: dict(BandSampler) sampler object to sample LSST-like conditions of each band
    :param zp_rms_frac_thresh: when scaling the zero point, the magnitude change must
        satisfy LSST_rms * zp_rms_frac_thresh > HSC_rms
    :param zp_max_mag_change: float maximum magnitude change allowed when changing zero point.
        If exceeded, the degradation fails.
    :param lsst_size_pix: int size of the output LSST-like images in pixels (square)
    :param bands: list of str bands corresponding to the images
    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param log_prefix: str prefix to add to log messages
    :return: success (bool), magnitude change applied (float)
    """
    success, degraded_images, mag_change = read_and_degrade(
        in_filename,
        dp0_sampler,
        zp_rms_frac_thresh,
        zp_max_mag_change,
        lsst_size_pix,
        bands,
        log_file,
        log_lock,
        log_prefix,
    )
    if success:
        write_degraded_images(
            out_filename,
            degraded_images,
            bands
        )
    return success, mag_change


def read_images(
        in_filename,
        bands='grizy',
        pix_scale=0.168
):
    """Read HSC images and PSFs from .fits files.

    :param in_filename: str path to input file (without band suffix)
    :param bands: list of str bands corresponding to the images
    :param pix_scale: float pixel scale to return if not found in header
    :return: list of images, list of PSFs (or None if not available), pixel scale in arcsec/pix
    """
    images = []
    psfs = []
    for b, band in enumerate(bands):
        image, hdr = fits.getdata(f"{in_filename}_img_{band}.fits", header=True)
        images.append(image)
        try:
            pix_scale = WCS(hdr).wcs.cd[1, 1] * 3600
        except AttributeError:
            pass
        if os.path.isfile(f"{in_filename}_psf_{band}.fits"):
            with fits.open(f"{in_filename}_psf_{band}.fits") as hdul:
                psfs.append(hdul[0].data)
        else:
            psfs.append(None)
    return images, psfs, pix_scale


def write_degraded_images(
        filename,
        images,
        bands='grizy'
):
    """Write degraded images to .fits files.

    :param filename: str path to output file (without band suffix)
    :param images: list of degraded images to save
    :param bands: list of str bands corresponding to the images
    :return: None
    """
    for b, band in enumerate(bands):
        hdu = fits.PrimaryHDU(data=images[b])
        hdu.writeto(f"{filename}_{band}.fits", overwrite=True)


def write_binary_images(
        filename,
        images,
        bands='grizy'
):
    """Write images to binary .fits files. Used for HSC raw data.

    :param filename: str path to output file (without band suffix)
    :param images: list of images to save
    :param bands: list of str bands corresponding to the images
    :return: None
    """
    for b, band in enumerate(bands):
        with open(f"{filename}_{band}.fits", 'wb') as f:
            f.write(images[b])


def log_message(log_file, log_lock, message, log_prefix=''):
    """Log a message to a log file with thread safety.

    :param log_file: str path to log file
    :param log_lock: Lock object to synchronize log file access
    :param message: str message to log
    :param log_prefix: str prefix to add to log messages
    :return: None
    """
    with log_lock:
        with open(log_file, 'a') as f:
            f.write(log_prefix + message + '\n')
