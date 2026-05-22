"""Integration test for legacy_to_rcs2.legacy_query.cutout.

Hits the real Legacy Survey viewer endpoint at one coordinate that is
known to be inside both Legacy DR10 and RCS2 footprints (the RCS2 field
0047G0 from /data/estudiantes/riugarte/rcs2/scripts/outputs, see header
OBJRA / OBJDEC).

Run as:
    python tests/test_cutout.py
or, with pytest installed:
    pytest tests/test_cutout.py
"""

import sys
import pathlib

import numpy as np

# Allow running directly with `python tests/test_cutout.py`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from legacy_to_rcs2.legacy_query.cutout import (
    query_legacy_cutout,
    LEGACY_NATIVE_PIXSCALE,
)


# Field 0047G0 (from RCS2 outputs) — equatorial, inside DECaLS coverage.
RA_DEG = 9.934792
DEC_DEG = -3.189194
BANDS = 'grz'
SIZE_PIX = 80   # 80 * 0.262 = 20.96 arcsec, leaves margin over 12 arcsec final


def test_query_legacy_cutout_returns_expected_shape():
    result = query_legacy_cutout(
        RA_DEG, DEC_DEG, bands=BANDS, size_pix=SIZE_PIX,
        pixscale=LEGACY_NATIVE_PIXSCALE,
    )

    assert set(result.keys()) == set(BANDS), \
        f"Expected bands {set(BANDS)}, got {set(result.keys())}"

    for band in BANDS:
        img = result[band]['image']
        hdr = result[band]['hdr']

        assert img.ndim == 2, f"Band {band}: image is not 2D (got ndim={img.ndim})"
        assert img.shape == (SIZE_PIX, SIZE_PIX), \
            f"Band {band}: shape {img.shape} != ({SIZE_PIX}, {SIZE_PIX})"
        assert img.dtype == np.float32, \
            f"Band {band}: dtype {img.dtype} != float32"
        assert np.isfinite(img).any(), f"Band {band}: no finite pixels"

        # WCS sanity: reference value should match the requested center
        # (to a few arcsec).
        assert abs(hdr['CRVAL1'] - RA_DEG) < 0.01, \
            f"Band {band}: CRVAL1={hdr['CRVAL1']} far from RA={RA_DEG}"
        assert abs(hdr['CRVAL2'] - DEC_DEG) < 0.01, \
            f"Band {band}: CRVAL2={hdr['CRVAL2']} far from Dec={DEC_DEG}"

        # NAXIS3 leakage check (we strip 3D-axis keys when slicing)
        assert 'NAXIS3' not in hdr, f"Band {band}: NAXIS3 not stripped"


def test_query_legacy_cutout_data_is_nontrivial():
    """The field is populated; flux must vary across the cutout."""
    result = query_legacy_cutout(
        RA_DEG, DEC_DEG, bands=BANDS, size_pix=SIZE_PIX,
        pixscale=LEGACY_NATIVE_PIXSCALE,
    )
    for band in BANDS:
        img = result[band]['image']
        assert np.nanstd(img) > 0, f"Band {band}: image is constant"


def _run_all_and_report():
    tests = [
        test_query_legacy_cutout_returns_expected_shape,
        test_query_legacy_cutout_data_is_nontrivial,
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
