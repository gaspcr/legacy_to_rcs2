"""Integration test for legacy_to_rcs2.rcs2_props.characterize_rcs2.

Runs the characterization over the sample of RCS2 frames at
/data/estudiantes/riugarte/rcs2/scripts/outputs and checks:
  - frames without PHOT_C are dropped (expected: most of the sample);
  - the output CSV is well-formed;
  - measured properties are physically plausible
    (seeing in [0.4, 2.0] arcsec, rms > 0, gain in [1.0, 2.0] e-/ADU).

Run as:
    python tests/test_characterize_rcs2.py
"""

import csv
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from legacy_to_rcs2.rcs2_props.characterize_rcs2 import (
    characterize_directory,
    extract_header_props,
)


SAMPLE_DIR = '/data/estudiantes/riugarte/rcs2/scripts/outputs'


def test_extract_header_props_filters_no_phot_c():
    """Frames without PHOT_C must be discarded (returns None)."""
    from astropy.io import fits
    import glob
    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, '*.fits')))
    assert len(files) > 0, "No sample frames found"

    has_phot, no_phot = 0, 0
    for f in files:
        with fits.open(f) as h:
            props = extract_header_props(h[0].header)
        if 'PHOT_C' in fits.getheader(f):
            has_phot += 1
            assert props is not None, f"{f}: dropped despite having PHOT_C"
            assert props['band'] in ('g', 'r', 'z')
            assert props['exp_time'] > 0
            assert 1.0 < props['gain'] < 2.0
        else:
            no_phot += 1
            assert props is None, f"{f}: kept despite missing PHOT_C"
    assert has_phot > 0 and no_phot > 0, (
        f"Expected both with/without PHOT_C in sample, got "
        f"has={has_phot}, no={no_phot}")


def test_characterize_directory_writes_valid_csv():
    """End-to-end: run on the real sample dir, verify CSV invariants."""
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, 'rcs2_props.csv')
        counts = characterize_directory(
            SAMPLE_DIR, out_csv, bands=('g', 'r', 'z'), verbose=False,
        )

        assert counts['processed'] > 0
        assert counts['ok'] > 0
        assert counts['ok'] < counts['processed'], \
            "Expected at least some discards (frames without PHOT_C)"

        with open(out_csv) as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == counts['ok']
        # All accepted bands
        bands = {r['band'] for r in rows}
        assert bands.issubset({'g', 'r', 'z'})

        for row in rows:
            seeing = float(row['seeing']) if row['seeing'] != '' else float('nan')
            rms = float(row['rms'])
            gain = float(row['gain'])
            exp_time = float(row['exp_time'])
            zp = float(row['zero_point'])

            # rms must be positive (image is not flat)
            assert rms > 0, f"{row['frame_id']}: non-positive rms"
            # gain on MegaCam is between 1.4 and 1.6 e-/ADU typically
            assert 1.0 < gain < 2.0, \
                f"{row['frame_id']}: gain={gain} out of MegaCam range"
            # exposure time positive
            assert exp_time > 0
            # PHOT_C is a magnitude, typically 24-27 for MegaCam
            assert 20 < zp < 30, \
                f"{row['frame_id']}: zp={zp} out of expected range"

            # Seeing is allowed to be NaN if too few stars were found
            # but if measured, must be sensible.
            import math
            if not math.isnan(seeing):
                assert 0.4 < seeing < 2.0, \
                    f"{row['frame_id']}: seeing={seeing} arcsec implausible"


def _run_all_and_report():
    tests = [
        test_extract_header_props_filters_no_phot_c,
        test_characterize_directory_writes_valid_csv,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all_and_report())
