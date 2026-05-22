#!/usr/bin/env python3
"""Curate a sample of RCS2 MegaCam CCD frames suitable for the
legacy_to_rcs2 properties sampler.

The RCS2 archive at /data/data1/RCS2/megacam/Patches/ contains thousands
of MegaCam exposures, each stored as a multi-extension Rice-compressed
FITS (.fz) with 36 CCD extensions. Many were taken under non-photometric
conditions and do not carry a PHOT_C zero point in their headers;
others have poor astrometric scatter. Running characterize_rcs2.py
directly on the full archive is slow and pollutes the sampler with
unreliable frames.

This script runs in two phases:

  1. SCAN -- walks the Patches tree, reading only the headers of each
     CCD extension (fast, no data decompression). Writes a CSV
     inventory with quality metrics per CCD: PHOT_C, CERROR, NASTRO,
     AIRMASS, EXPTIME, GAIN.

  2. EXTRACT -- filters the inventory by quality cuts, optionally
     subsamples N CCDs per band with a fixed random seed, then calls
     funpack to extract each selected extension as a flat FITS file
     named <pointing>_<band>_<ext>.fits in the output directory.

Either phase can be run alone (--scan-only / --extract-only) provided
a pre-existing inventory CSV is given.

Default quality cuts (per the RCS2 strong-lensing-search use case):

  PHOT_C  present       photometric calibration available
  CERROR  < 1.0 arcsec  astrometric scatter
  NASTRO  > 20          number of stars used in astrometry
  AIRMASS < 1.5         reasonable atmospheric conditions

These can be overridden on the command line.

Output naming convention follows makestamps_rcs2.py for compatibility
with the user's existing RCS2 tooling.

Example
-------
Run both phases at once, producing a curated sample of ~50 CCDs per
band in /data/estudiantes/riugarte/rcs2/rcs2_props_sample/:

    python scripts/select_rcs2_sample.py \\
        /data/data1/RCS2/megacam/Patches/ \\
        /data/estudiantes/riugarte/rcs2/rcs2_props_sample/ \\
        --inventory inventory.csv \\
        --n-per-band 50
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from collections import Counter

from astropy.io import fits


# ----------------------------------------------------------------------
# Default paths / constants (match makestamps_rcs2.py)
# ----------------------------------------------------------------------

DEFAULT_IMAGES_ROOT = "/data/data1/RCS2/megacam/Patches"

# .fz filenames look like <pointing>_<band>.fz or ss<pointing>_<band>.fz
# (the ss prefix is used for some reduced z-band files in MegaCam).
RCS2_BANDS = ('g', 'r', 'i', 'z')

# Header keys used for the inventory. Keep this list explicit so the
# CSV schema is auditable.
INVENTORY_FIELDS = [
    'fz_path', 'ext', 'pointing', 'band', 'ccdname',
    'phot_c', 'cerror', 'nastro', 'airmass',
    'exp_time', 'gain', 'saturate',
]


# ----------------------------------------------------------------------
# Phase 1: SCAN
# ----------------------------------------------------------------------


def scan_inventory(images_root, output_csv, bands=RCS2_BANDS,
                   max_pointings=None, verbose=True):
    """Walk Patches/ and write a CSV with per-CCD header metrics.

    No file data is decompressed; only headers are read.
    """
    pointings = list(_iter_pointings(images_root))
    if max_pointings is not None:
        pointings = pointings[:max_pointings]

    n_total_ccds = 0
    n_files = 0
    with open(output_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()

        for i, (patch, pointing, dirpath) in enumerate(pointings, 1):
            if verbose:
                print(f"[{i}/{len(pointings)}] {patch}/{pointing}")
            for fz_name in os.listdir(dirpath):
                if not fz_name.endswith('.fz'):
                    continue
                band = _band_from_filename(fz_name)
                if band is None or band not in bands:
                    continue
                fz_path = os.path.join(dirpath, fz_name)
                try:
                    rows = _scan_fz(fz_path, pointing, band)
                except Exception as e:
                    print(f"  WARN: {fz_name}: {e}")
                    continue
                writer.writerows(rows)
                n_total_ccds += len(rows)
                n_files += 1

    if verbose:
        print(f"\nScanned {n_files} .fz files, "
              f"{n_total_ccds} CCD extensions -> {output_csv}")


def _iter_pointings(root):
    """Yield (patch, pointing, dirpath) for every pointing directory."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Images root not found: {root}")
    for patch in sorted(os.listdir(root)):
        patch_dir = os.path.join(root, patch)
        if not os.path.isdir(patch_dir):
            continue
        for pointing in sorted(os.listdir(patch_dir)):
            pointing_dir = os.path.join(patch_dir, pointing)
            if os.path.isdir(pointing_dir):
                yield patch, pointing, pointing_dir


def _band_from_filename(fz_name):
    """Parse band letter from filename: <pointing>_<band>.fz or
    ss<pointing>_<band>.fz . Returns None on no match."""
    base = fz_name[:-3] if fz_name.endswith('.fz') else fz_name
    parts = base.rsplit('_', 1)
    if len(parts) != 2:
        return None
    band = parts[1]
    return band if band in 'griz' else None


def _scan_fz(fz_path, pointing, band):
    """Read all CCD extension headers from a .fz file. Returns one
    inventory row per CCD."""
    rows = []
    with fits.open(fz_path) as hdul:
        # HDU[0] is metadata-only; HDU[1..N] are the CCD extensions.
        for ext in range(1, len(hdul)):
            hdr = hdul[ext].header
            rows.append({
                'fz_path': fz_path,
                'ext': f"{ext - 1:02d}",       # ccd00..ccd35 convention
                'pointing': pointing,
                'band': band,
                'ccdname': hdr.get('EXTNAME', ''),
                'phot_c': hdr.get('PHOT_C', ''),
                'cerror': hdr.get('CERROR', ''),
                'nastro': hdr.get('NASTRO', ''),
                'airmass': hdr.get('AIRMASS', ''),
                'exp_time': hdr.get('EXPTIME', ''),
                'gain': hdr.get('GAIN', ''),
                'saturate': hdr.get('SATURATE', ''),
            })
    return rows


# ----------------------------------------------------------------------
# Phase 2: EXTRACT
# ----------------------------------------------------------------------


def filter_and_extract(inventory_csv, output_dir,
                        phot_c_required=True,
                        cerror_max=1.0,
                        nastro_min=20,
                        airmass_max=1.5,
                        bands=RCS2_BANDS,
                        n_per_band=None,
                        seed=42,
                        funpack_cmd='funpack',
                        skip_existing=True,
                        verbose=True):
    """Filter the inventory by quality cuts and funpack selected CCDs."""
    selected = _apply_filters(
        inventory_csv, bands, phot_c_required, cerror_max,
        nastro_min, airmass_max, verbose,
    )

    if n_per_band is not None:
        selected = _subsample_per_band(selected, n_per_band, seed)
        if verbose:
            print(f"Subsampled to {n_per_band} per band (seed={seed}):")
            for band, n in sorted(Counter(r['band'] for r in selected).items()):
                print(f"  {band}: {n}")

    os.makedirs(output_dir, exist_ok=True)
    n_done, n_skip, n_fail = 0, 0, 0
    for i, row in enumerate(selected, 1):
        out_name = f"{row['pointing']}_{row['band']}_{row['ext']}.fits"
        out_path = os.path.join(output_dir, out_name)
        if skip_existing and os.path.isfile(out_path):
            n_skip += 1
            continue
        if verbose:
            print(f"[{i}/{len(selected)}] funpack {out_name}")
        try:
            _funpack_extension(funpack_cmd, row['fz_path'],
                               row['ext'], out_path)
            n_done += 1
        except subprocess.CalledProcessError as e:
            print(f"  WARN: funpack failed: {e}")
            n_fail += 1

    if verbose:
        print(f"\nExtracted {n_done} new, skipped {n_skip} existing, "
              f"failed {n_fail}")


def _apply_filters(inventory_csv, bands, phot_c_required,
                    cerror_max, nastro_min, airmass_max, verbose):
    """Read inventory CSV and return the rows that pass all quality cuts."""
    selected = []
    n_total = 0
    n_reject = Counter()
    with open(inventory_csv) as fh:
        for row in csv.DictReader(fh):
            n_total += 1
            if row['band'] not in bands:
                n_reject['band'] += 1
                continue
            if phot_c_required and not _is_number(row['phot_c']):
                n_reject['no_phot_c'] += 1
                continue
            if not _is_number(row['cerror']) or float(row['cerror']) >= cerror_max:
                n_reject['cerror'] += 1
                continue
            if not _is_number(row['nastro']) or int(float(row['nastro'])) <= nastro_min:
                n_reject['nastro'] += 1
                continue
            if not _is_number(row['airmass']) or float(row['airmass']) >= airmass_max:
                n_reject['airmass'] += 1
                continue
            selected.append(row)

    if verbose:
        print(f"\nFilter: {n_total} rows -> {len(selected)} pass")
        for reason, n in sorted(n_reject.items()):
            print(f"  rejected by {reason}: {n}")
    return selected


def _subsample_per_band(rows, n_per_band, seed):
    """Random sample up to n_per_band rows per band, with fixed seed."""
    rng = random.Random(seed)
    by_band = {}
    for r in rows:
        by_band.setdefault(r['band'], []).append(r)
    out = []
    for band, band_rows in by_band.items():
        if len(band_rows) > n_per_band:
            band_rows = rng.sample(band_rows, n_per_band)
        out.extend(band_rows)
    return out


def _funpack_extension(funpack_cmd, fz_path, ext, out_path):
    """Run funpack to extract a single CCD extension to a flat FITS."""
    subprocess.run(
        [funpack_cmd, '-O', out_path, '-E', ext, fz_path],
        check=True, capture_output=True,
    )


def _is_number(s):
    if s is None or s == '':
        return False
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Curate a sample of RCS2 CCD frames for the props sampler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('images_root', nargs='?', default=DEFAULT_IMAGES_ROOT,
                   help="Root of the RCS2 patches tree (default: %(default)s)")
    p.add_argument('output_dir',
                   help="Where to write extracted .fits files")
    p.add_argument('--inventory', default='rcs2_inventory.csv',
                   help="Inventory CSV path (read or written depending on mode)")
    p.add_argument('--scan-only', action='store_true',
                   help="Only scan headers and write inventory, do not extract")
    p.add_argument('--extract-only', action='store_true',
                   help="Skip scan; use existing inventory CSV")
    p.add_argument('--bands', default='grz',
                   help="Bands to consider (default 'grz')")
    p.add_argument('--n-per-band', type=int, default=None,
                   help="Random subsample size per band (default: keep all)")
    p.add_argument('--seed', type=int, default=42,
                   help="RNG seed for subsampling")
    p.add_argument('--cerror-max', type=float, default=1.0,
                   help="Max astrometric scatter in arcsec (default 1.0)")
    p.add_argument('--nastro-min', type=int, default=20,
                   help="Min astrometry stars (default 20)")
    p.add_argument('--airmass-max', type=float, default=1.5,
                   help="Max airmass (default 1.5)")
    p.add_argument('--no-phot-c-required', action='store_true',
                   help="Do NOT require PHOT_C (use only for debug)")
    p.add_argument('--max-pointings', type=int, default=None,
                   help="Limit scan to N pointings (debug aid)")
    p.add_argument('--funpack-cmd', default='funpack',
                   help="Path to funpack executable (default: search PATH)")
    p.add_argument('--no-skip-existing', action='store_true',
                   help="Re-extract files that already exist in output_dir")
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    bands = tuple(args.bands)
    verbose = not args.quiet

    if not args.extract_only:
        scan_inventory(args.images_root, args.inventory,
                        bands=bands, max_pointings=args.max_pointings,
                        verbose=verbose)

    if not args.scan_only:
        filter_and_extract(
            args.inventory, args.output_dir,
            phot_c_required=not args.no_phot_c_required,
            cerror_max=args.cerror_max,
            nastro_min=args.nastro_min,
            airmass_max=args.airmass_max,
            bands=bands,
            n_per_band=args.n_per_band,
            seed=args.seed,
            funpack_cmd=args.funpack_cmd,
            skip_existing=not args.no_skip_existing,
            verbose=verbose,
        )


if __name__ == '__main__':
    sys.exit(main())
