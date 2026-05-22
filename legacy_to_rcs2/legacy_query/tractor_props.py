"""Query per-band PSF FWHM and per-pixel RMS at a coordinate from the
Legacy Survey DR10 Tractor catalog via the Astro Data Lab TAP service.

Background
----------
ls_dr10.tractor provides per-object columns:
  - psfsize_{band}  : weighted-average PSF FWHM at the object position,
                      in arcsec, on the coadd image.
  - psfdepth_{band} : 5-sigma point-source detection limit, defined so
                      that the 5-sigma flux limit is 5/sqrt(psfdepth)
                      in nanomaggies.

We use the NEAREST tractor object to (ra, dec) as a proxy for the local
PSF and noise properties. The PSF varies smoothly over a brick
(0.25 deg side), so for sub-arcmin offsets this is accurate.

Per-pixel RMS derivation
------------------------
For optimal PSF photometry on a Gaussian PSF, the flux variance is
    sigma_F^2 = sigma_pixel^2 / sum_i P_i^2 = sigma_pixel^2 * Neff
where Neff = 4*pi*sigma^2 (pixels), sigma = FWHM/(2.355*pixscale).
The 5-sigma flux limit is 5*sigma_F, and psfdepth is defined so that
this limit equals 5/sqrt(psfdepth). Therefore
    sigma_pixel = 1 / sqrt(psfdepth * Neff)

This gives the per-pixel noise RMS in nanomaggies (the native flux unit
of Legacy Survey coadds, ZP=22.5). It is consistent with sigma measured
on the cutout itself to ~10-20% (validated empirically at field 0047G0).
"""

import math
import time

import numpy as np
import requests
from astropy.io import ascii as astropy_ascii


DATALAB_TAP_SYNC_URL = "https://datalab.noirlab.edu/tap/sync"

# Native DECam pixel scale (arcsec/pixel) used by Legacy DR10 coadds.
# Needed to convert psfsize (arcsec) -> sigma in pixels for Neff.
LEGACY_NATIVE_PIXSCALE = 0.262

# Gaussian FWHM -> sigma factor: sigma = FWHM / GAUSSIAN_FWHM_TO_SIGMA
GAUSSIAN_FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # ~2.3548


def query_tractor_props(
    ra,
    dec,
    bands='grz',
    search_radius_arcsec=60.0,
    pixscale=LEGACY_NATIVE_PIXSCALE,
    timeout=60,
    max_retries=3,
    retry_wait=2.0,
):
    """Return PSF FWHM and per-pixel RMS at (ra, dec) for the requested bands.

    Picks the tractor object NEAREST to (ra, dec) within
    `search_radius_arcsec` and returns its psfsize / psfdepth values.

    :param ra: float, RA in degrees (J2000)
    :param dec: float, Dec in degrees (J2000)
    :param bands: str, concatenated band letters (default 'grz')
    :param search_radius_arcsec: float, cone radius used to find the
        nearest tractor object. 60" is small enough to be fast and large
        enough that an object is essentially always found in DR10.
    :param pixscale: float, Legacy pixel scale (arcsec/pixel). Default is
        the native DECam value (0.262). Used to derive RMS per pixel.
    :param timeout: float, HTTP timeout in seconds
    :param max_retries: int, HTTP retries on transient failure
    :param retry_wait: float, linear backoff in seconds between retries
    :return: dict {band: {'fwhm_arcsec': float, 'rms_pixel': float,
        'psfdepth': float, 'dist_arcsec': float}}
    :raises RuntimeError: if no tractor object found within search radius
    :raises requests.HTTPError: on persistent HTTP failure
    """
    row = _query_nearest_tractor_row(
        ra, dec, bands, search_radius_arcsec,
        timeout, max_retries, retry_wait,
    )

    out = {}
    for band in bands:
        fwhm = float(row[f'psfsize_{band}'])
        psfdepth = float(row[f'psfdepth_{band}'])
        rms_pix = _rms_pixel_from_psfdepth(psfdepth, fwhm, pixscale)
        out[band] = {
            'fwhm_arcsec': fwhm,
            'rms_pixel': rms_pix,
            'psfdepth': psfdepth,
            'dist_arcsec': float(row['dist_arcsec']),
        }
    return out


def _rms_pixel_from_psfdepth(psfdepth, fwhm_arcsec, pixscale):
    """Convert (psfdepth, FWHM) to per-pixel RMS in nanomaggies.

    Returns NaN if psfdepth <= 0 or FWHM is non-positive (edge cases at
    brick boundaries where the catalog has no valid measurement).
    """
    if not (psfdepth > 0 and fwhm_arcsec > 0):
        return float('nan')
    sigma_pix = fwhm_arcsec / (GAUSSIAN_FWHM_TO_SIGMA * pixscale)
    neff = 4.0 * math.pi * sigma_pix * sigma_pix
    return 1.0 / math.sqrt(psfdepth * neff)


def _query_nearest_tractor_row(ra, dec, bands, radius_arcsec,
                                timeout, max_retries, retry_wait):
    """Run the ADQL TAP query and return the closest row as a dict."""
    radius_deg = radius_arcsec / 3600.0
    cos_dec = max(math.cos(math.radians(dec)), 0.1)
    d_ra = radius_deg / cos_dec

    band_cols = ", ".join(
        f"psfsize_{b}, psfdepth_{b}" for b in bands
    )

    # The Astro Data Lab TAP ADQL parser rejects POINT(...) outside CONTAINS
    # and complex expressions in ORDER BY, so we use a simple RA/Dec
    # bounding box for the WHERE filter and q3c_dist as a scalar for
    # ordering. q3c is the spatial index used internally.
    adql = (
        f"SELECT TOP 1 {band_cols}, "
        f"q3c_dist(ra, dec, {ra}, {dec}) * 3600.0 AS dist_arcsec "
        f"FROM ls_dr10.tractor "
        f"WHERE ra BETWEEN {ra - d_ra} AND {ra + d_ra} "
        f"  AND dec BETWEEN {dec - radius_deg} AND {dec + radius_deg} "
        f"ORDER BY dist_arcsec ASC"
    )

    response = _tap_sync(adql, timeout, max_retries, retry_wait)

    table = astropy_ascii.read(response.text, format='csv')

    if len(table) == 0:
        raise RuntimeError(
            f"No ls_dr10.tractor object found within {radius_arcsec}\" of "
            f"(ra={ra}, dec={dec}). The position may be outside Legacy "
            f"DR10 footprint, or in an underdense region."
        )

    return table[0]


def _tap_sync(adql, timeout, max_retries, retry_wait):
    """POST an ADQL query to the Data Lab TAP sync endpoint, with retry."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                DATALAB_TAP_SYNC_URL,
                data={
                    'REQUEST': 'doQuery',
                    'LANG': 'ADQL',
                    'FORMAT': 'csv',
                    'QUERY': adql,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            # The TAP service returns 200 even on ADQL errors; the body
            # then starts with a VOTable XML error envelope instead of CSV.
            if r.text.lstrip().startswith('<?xml'):
                raise RuntimeError(f"TAP ADQL error:\n{r.text[:500]}")
            return r
        except (requests.RequestException, RuntimeError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(retry_wait * (attempt + 1))
    raise last_exc
