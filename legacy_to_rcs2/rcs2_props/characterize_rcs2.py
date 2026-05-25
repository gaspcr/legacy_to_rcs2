"""Characterize RCS2 MegaCam single-exposure CCD frames into a properties
table used by the RCS2 sampler at degradation time.

For each input FITS frame we extract from the header:
  - band             from FILTER (e.g. 'g.MP9401' -> 'g')
  - exp_time         from EXPTIME (seconds)
  - gain             from GAIN (e-/ADU; varies per amp, ~1.4-1.6)
  - zero_point       from PHOT_C (single-exposure AB magnitude ZP).
                     Frames without PHOT_C are DISCARDED -- they come
                     from non-photometric nights and would inject bias
                     if we imputed a band-average value.

And we measure on the image:
  - median, rms      via photutils-based iterative background estimate
                     with source masking (legacy_to_rcs2.utils).
                     Measured on a central crop to avoid CCD edge
                     artefacts (overscan residuals, bad columns).
  - seeing           median FWHM (arcsec) of bright unsaturated stars
                     detected by DAOStarFinder and fit with a Gaussian2D
                     model. RCS2 headers do not carry a SEEING keyword.

Output is a CSV with one row per accepted frame and columns:
  band, frame_id, exp_time, gain, zero_point, seeing, rms, median, n_stars

This file is consumed by legacy_to_rcs2.rcs2_props.rcs2_samplers to
produce per-band samplers at degradation time.
"""

import argparse
import csv
import logging
import os
import re
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from legacy_to_rcs2.utils import photutils_background_iterative


# RCS2 / MegaCam constants -- documented so the assumptions are auditable.
RCS2_PIX_SCALE_ARCSEC = 0.185
RCS2_BANDS = ('g', 'r', 'z')
FILTER_PATTERN = re.compile(r'^([griz])\.')   # 'g.MP9401' -> 'g'

# Central crop used for noise estimation. The full CCD is 2112x4644;
# a 1500x1500 central region is large enough for robust statistics and
# small enough to avoid edge defects.
NOISE_CROP_PIX = 1500

# Seeing measurement parameters
SEEING_FWHM_GUESS_ARCSEC = 1.0           # initial guess for DAOStarFinder
SEEING_THRESHOLD_NSIGMA = 20.0           # only bright sources for stars
SEEING_MIN_STARS = 5                     # below this, mark seeing as NaN
SEEING_MAX_CANDIDATES = 200              # cap for runtime
SEEING_FIT_BOX_HALF = 10                 # half-size of stamp for Gaussian fit
SEEING_PEAK_MAX_FRAC = 0.5               # stars with peak > frac*SATURATE rejected
SEEING_FWHM_RANGE_ARCSEC = (0.3, 3.0)    # accept fits within this range


# ----------------------------------------------------------------------
# Header parsing
# ----------------------------------------------------------------------


def classify_header(header):
    """Parse the header once and decide whether the frame is usable.

    Returns (props, reason) where exactly one is non-None:
      - props is the dict of needed keys when the frame is accepted
        (reason is then None);
      - props is None and reason is the discard code otherwise, one of
        'band'      (FILTER missing/unparseable or not a grz band) or
        'no_phot_c' (no PHOT_C -> non-photometric night, discarded
                     rather than imputed to avoid biasing the sampler).

    The band check precedes the PHOT_C check, matching the order the
    cuts are applied, so the reason reflects the first failing condition.
    """
    filt = header.get('FILTER', '')
    m = FILTER_PATTERN.match(filt)
    if m is None or m.group(1) not in RCS2_BANDS:
        return None, 'band'

    phot_c = header.get('PHOT_C', None)
    if phot_c is None:
        return None, 'no_phot_c'

    return {
        'band': m.group(1),
        'exp_time': float(header['EXPTIME']),
        'gain': float(header['GAIN']),
        'zero_point': float(phot_c),
        'saturate': float(header.get('SATURATE', 65535.0)),
    }, None


def extract_header_props(header):
    """Backward-compatible wrapper: return the props dict, or None if the
    frame must be discarded. New code should call classify_header() to
    also obtain the discard reason for logging."""
    props, _ = classify_header(header)
    return props


# ----------------------------------------------------------------------
# Noise (median, rms)
# ----------------------------------------------------------------------


def measure_noise(image):
    """Median sky level and per-pixel RMS via iterative source masking.

    Operates on a central crop to avoid edge artefacts. Returns
    (median, rms) in image units (ADU on RCS2 Elixir frames).

    The iterative function inherited from the upstream package assumes
    a sky-subtracted image (init_median=0). RCS2 Elixir frames are NOT
    sky-subtracted (sky ~ 2000 ADU), so we pre-compute a robust initial
    median via sigma_clipped_stats and pass it in. Without this, the
    very first detect_threshold ends up masking 99% of the image and
    the iteration returns garbage.
    """
    crop = _central_crop(image, NOISE_CROP_PIX)
    _, init_median, init_rms = sigma_clipped_stats(crop, sigma=3.0, maxiters=5)
    median, rms, _ = photutils_background_iterative(
        crop, init_median=float(init_median), init_rms=float(init_rms),
    )
    return float(median), float(rms)


def _central_crop(image, side_pix):
    """Return a centered square crop of `side_pix` from a 2D image."""
    ny, nx = image.shape
    side = min(side_pix, ny, nx)
    y0 = (ny - side) // 2
    x0 = (nx - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


# ----------------------------------------------------------------------
# Seeing (FWHM via star fitting)
# ----------------------------------------------------------------------


def measure_seeing(image, saturate, pix_scale=RCS2_PIX_SCALE_ARCSEC):
    """Estimate seeing FWHM in arcsec by detecting bright unsaturated
    stars and fitting a 2D Gaussian to each. Returns (fwhm_arcsec, n_used).

    Returns (NaN, 0) if fewer than SEEING_MIN_STARS stars are usable.
    """
    crop = _central_crop(image, NOISE_CROP_PIX)
    _, median, std = sigma_clipped_stats(crop, sigma=3.0, maxiters=5)

    fwhm_guess_pix = SEEING_FWHM_GUESS_ARCSEC / pix_scale
    finder = DAOStarFinder(
        fwhm=fwhm_guess_pix,
        threshold=SEEING_THRESHOLD_NSIGMA * std,
        exclude_border=True,
    )
    sources = finder(crop - median)
    if sources is None or len(sources) == 0:
        return float('nan'), 0

    sources = sources[sources['peak'] < SEEING_PEAK_MAX_FRAC * saturate]
    sources.sort('peak')
    sources.reverse()                                # brightest first
    sources = sources[:SEEING_MAX_CANDIDATES]

    fwhm_list = []
    for src in sources:
        fwhm_pix = _fit_star_fwhm_pix(
            crop, int(round(src['xcentroid'])), int(round(src['ycentroid'])),
            median,
        )
        if fwhm_pix is None:
            continue
        fwhm_arcsec = fwhm_pix * pix_scale
        if SEEING_FWHM_RANGE_ARCSEC[0] < fwhm_arcsec < SEEING_FWHM_RANGE_ARCSEC[1]:
            fwhm_list.append(fwhm_arcsec)

    if len(fwhm_list) < SEEING_MIN_STARS:
        return float('nan'), len(fwhm_list)
    return float(np.median(fwhm_list)), len(fwhm_list)


def _fit_star_fwhm_pix(image, x, y, sky_median):
    """Fit a circular 2D Gaussian to a stamp around (x, y). Returns
    FWHM in pixels, or None if the fit is invalid."""
    half = SEEING_FIT_BOX_HALF
    ny, nx = image.shape
    if not (half <= x < nx - half and half <= y < ny - half):
        return None
    stamp = image[y - half:y + half + 1, x - half:x + half + 1] - sky_median

    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    g_init = models.Gaussian2D(
        amplitude=stamp.max(),
        x_mean=half, y_mean=half,
        x_stddev=2.0, y_stddev=2.0,
    )
    g_init.x_stddev.tied = lambda model: model.y_stddev   # circular
    fitter = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            g_fit = fitter(g_init, xx, yy, stamp)
        except Exception:
            return None

    sigma = float(g_fit.y_stddev.value)
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))      # FWHM in pixels


# ----------------------------------------------------------------------
# Top-level: iterate over a directory
# ----------------------------------------------------------------------


def characterize_directory(input_dir, output_csv, bands=RCS2_BANDS,
                            max_files=None, verbose=True):
    """Iterate FITS files under `input_dir`, build the props CSV.

    Returns a dict with counts: {processed, discarded_no_phot_c,
    discarded_band, discarded_io, ok}.
    """
    fits_files = sorted(_iter_fits(input_dir))
    if max_files is not None:
        fits_files = fits_files[:max_files]

    counts = {'processed': len(fits_files), 'discarded_no_phot_c': 0,
              'discarded_band': 0, 'discarded_io': 0, 'ok': 0}
    # Map each discard reason code from _process_one to its counter key.
    reason_to_counter = {
        'no_phot_c': 'discarded_no_phot_c',
        'band': 'discarded_band',
        'io': 'discarded_io',
    }

    fieldnames = ['band', 'frame_id', 'exp_time', 'gain', 'zero_point',
                  'seeing', 'rms', 'median', 'n_stars']

    with open(output_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(fits_files, 1):
            frame_id = os.path.splitext(os.path.basename(path))[0]
            if verbose:
                print(f"[{i}/{len(fits_files)}] {frame_id}")

            row, reason = _process_one(path, frame_id, bands)
            if row is None:
                counts[reason_to_counter[reason]] += 1
                if verbose:
                    print(f"    discarded ({reason})")
                continue

            writer.writerow(row)
            counts['ok'] += 1

    _print_summary(counts, output_csv)
    return counts


def _process_one(path, frame_id, bands):
    """Return (row, reason): row is the props dict when accepted (reason
    None), otherwise row is None and reason is the discard code
    ('io', 'band', 'no_phot_c')."""
    try:
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            image = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception as e:
        logging.warning(f"{frame_id}: read failed: {e}")
        return None, 'io'

    hdr_props, reason = classify_header(hdr)
    if hdr_props is None:
        return None, reason
    if hdr_props['band'] not in bands:
        return None, 'band'   # excluded by the --bands selection

    median, rms = measure_noise(image)
    seeing, n_stars = measure_seeing(image, hdr_props['saturate'])

    return {
        'band': hdr_props['band'],
        'frame_id': frame_id,
        'exp_time': hdr_props['exp_time'],
        'gain': hdr_props['gain'],
        'zero_point': hdr_props['zero_point'],
        'seeing': seeing,
        'rms': rms,
        'median': median,
        'n_stars': n_stars,
    }, None


def _iter_fits(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith('.fits'):
                yield os.path.join(dirpath, name)


def _print_summary(counts, output_csv):
    n = counts['processed']
    ok = counts['ok']
    print(f"\nDone. {ok}/{n} frames written to {output_csv}")
    if n > 0 and ok < n:
        print(f"Discarded {n - ok} frames:")
        print(f"  no PHOT_C (non-photometric): {counts['discarded_no_phot_c']}")
        print(f"  unsupported/excluded band  : {counts['discarded_band']}")
        print(f"  read / IO error            : {counts['discarded_io']}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=(
        "Build the RCS2 properties CSV used by the legacy_to_rcs2 sampler."))
    p.add_argument('input_dir',
                   help="Directory tree containing RCS2 FITS frames")
    p.add_argument('output_csv',
                   help="Path for the resulting properties CSV")
    p.add_argument('--bands', default='grz',
                   help="Bands to include (default 'grz')")
    p.add_argument('--max-files', type=int, default=None,
                   help="Process at most this many files (debug aid)")
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    characterize_directory(
        args.input_dir,
        args.output_csv,
        bands=tuple(args.bands),
        max_files=args.max_files,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
