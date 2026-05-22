"""Query multi-band image cutouts from the Legacy Survey viewer.

The Legacy Survey viewer cutout endpoint at
    https://www.legacysurvey.org/viewer/cutout.fits
accepts (ra, dec, layer, pixscale, bands, size) and returns a FITS file
with a 3D image cube of shape (n_bands, ny, nx). The band order in the
cube matches the order of the `bands` query parameter.

For our pipeline (Legacy DR10 -> RCS2 degradation):
  - layer is 'ls-dr10'
  - bands is 'grz'
  - pixscale=0.262 (native DECam) so the service does NOT resample;
    the resampling to RCS2 pixscale (0.185) is done downstream by
    legacy_to_rcs2.data_degradation.resample.resample_image with
    flux conservation.
"""

import io
import time

import numpy as np
import requests
from astropy.io import fits


LEGACY_VIEWER_CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# Native DECam pixel scale used by Legacy Survey DR10 coadds.
# Requesting this value ensures the viewer does NOT resample server-side.
LEGACY_NATIVE_PIXSCALE = 0.262


def query_legacy_cutout(
    ra,
    dec,
    bands='grz',
    layer='ls-dr10',
    pixscale=LEGACY_NATIVE_PIXSCALE,
    size_pix=80,
    timeout=60,
    max_retries=3,
    retry_wait=2.0,
):
    """Download a multi-band Legacy Survey cutout.

    :param ra: float, right ascension in degrees (J2000)
    :param dec: float, declination in degrees (J2000)
    :param bands: str, concatenated band letters (default 'grz')
    :param layer: str, Legacy Survey layer name (default 'ls-dr10')
    :param pixscale: float, pixel scale in arcsec/pixel.
        Default is the DECam native value 0.262, which avoids
        server-side resampling. Any other value triggers resampling.
    :param size_pix: int, output cutout side length in pixels.
        Choose so that size_pix * pixscale exceeds the final RCS2 stamp
        size (12 arcsec) with margin for PSF convolution and edge drop.
    :param timeout: float, per-request HTTP timeout in seconds
    :param max_retries: int, number of HTTP retries on transient failure
    :param retry_wait: float, base seconds between retries (linear backoff)
    :return: dict mapping each requested band -> {'image': 2D float32 ndarray,
        'hdr': astropy.io.fits.Header} with the 2D per-band header.
    :raises requests.HTTPError: on persistent HTTP failure
    :raises IOError: on empty or malformed FITS response
    :raises KeyError: if a requested band is missing from the response
    """
    params = {
        'ra': float(ra),
        'dec': float(dec),
        'layer': layer,
        'pixscale': float(pixscale),
        'bands': bands,
        'size': int(size_pix),
    }

    response = _http_get_with_retry(
        LEGACY_VIEWER_CUTOUT_URL, params, timeout, max_retries, retry_wait
    )

    return _parse_cutout_response(response.content, bands)


def _http_get_with_retry(url, params, timeout, max_retries, retry_wait):
    """GET with linear backoff. Re-raises the last exception on exhaustion."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(retry_wait * (attempt + 1))
    raise last_exc


def _parse_cutout_response(content_bytes, requested_bands):
    """Parse the FITS bytes returned by the viewer into a per-band dict.

    The viewer returns the cutout as a 3D primary HDU with shape
    (n_bands, ny, nx). For a single-band request the data may be 2D.
    """
    with fits.open(io.BytesIO(content_bytes)) as hdul:
        hdu = hdul[0]
        data = hdu.data
        hdr = hdu.header.copy()

    if data is None:
        raise IOError("Legacy cutout returned an empty primary HDU")

    if data.ndim == 2:
        planes = [data]
    elif data.ndim == 3:
        planes = [data[i] for i in range(data.shape[0])]
    else:
        raise IOError(f"Unexpected FITS data ndim={data.ndim} (shape={data.shape})")

    if len(planes) != len(requested_bands):
        raise IOError(
            f"Expected {len(requested_bands)} bands, got {len(planes)} planes"
        )

    band_hdr = _strip_third_axis_keys(hdr)

    out = {}
    for band, plane in zip(requested_bands, planes):
        out[band] = {
            'image': np.asarray(plane, dtype=np.float32),
            'hdr': band_hdr.copy(),
        }

    for b in requested_bands:
        if b not in out:
            raise KeyError(f"Band {b} missing in cutout response (got {list(out)})")

    return out


def _strip_third_axis_keys(hdr):
    """Drop NAXIS3 / CRVAL3 / CTYPE3 etc. so the header is valid for 2D data."""
    h = hdr.copy()
    for key in list(h.keys()):
        if key in ('NAXIS3',) or (key and key[-1] == '3' and key.startswith(
            ('CRVAL', 'CRPIX', 'CTYPE', 'CDELT', 'CUNIT', 'CD1_', 'CD2_', 'CD3_',
             'PC1_', 'PC2_', 'PC3_')
        )):
            del h[key]
    h['NAXIS'] = 2
    return h
