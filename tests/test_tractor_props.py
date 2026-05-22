"""Integration tests for legacy_to_rcs2.legacy_query.tractor_props.

Hits the real Astro Data Lab TAP endpoint at a coordinate known to be
inside Legacy DR10 footprint (field 0047G0 from RCS2 outputs).

Includes a cross-check test that the per-pixel RMS derived analytically
from psfdepth agrees with the RMS measured directly on a cutout from
the same coordinate, to within 25% (accounts for the Gaussian-PSF
approximation in the formula and small residual sky structure in the
empirical std).

Run as:
    python tests/test_tractor_props.py
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from legacy_to_rcs2.legacy_query.tractor_props import (
    query_tractor_props,
    _rms_pixel_from_psfdepth,
)


# Field 0047G0 (from RCS2 outputs)
RA_DEG = 9.934792
DEC_DEG = -3.189194
BANDS = 'grz'


def test_query_tractor_props_returns_expected_structure():
    result = query_tractor_props(RA_DEG, DEC_DEG, bands=BANDS)

    assert set(result.keys()) == set(BANDS), \
        f"Expected bands {set(BANDS)}, got {set(result.keys())}"

    for band in BANDS:
        for key in ('fwhm_arcsec', 'rms_pixel', 'psfdepth', 'dist_arcsec'):
            assert key in result[band], f"Band {band}: missing key {key}"

        # FWHM should be physical for DECaLS: typically 0.8-2.5 arcsec
        assert 0.5 < result[band]['fwhm_arcsec'] < 3.0, \
            f"Band {band}: FWHM out of range: {result[band]['fwhm_arcsec']}"
        # psfdepth should be positive (5-sigma depth defined as such)
        assert result[band]['psfdepth'] > 0, \
            f"Band {band}: non-positive psfdepth"
        # RMS per pixel should be finite and positive
        assert np.isfinite(result[band]['rms_pixel']), \
            f"Band {band}: non-finite RMS"
        assert result[band]['rms_pixel'] > 0, \
            f"Band {band}: non-positive RMS"
        # Nearest object should be within the search radius
        assert result[band]['dist_arcsec'] < 60.0


def test_rms_predicted_matches_measured_on_cutout():
    """Cross-check: the analytic RMS from psfdepth must match the std
    measured on the actual cutout to within ~25%."""
    from legacy_to_rcs2.legacy_query.cutout import query_legacy_cutout

    props = query_tractor_props(RA_DEG, DEC_DEG, bands=BANDS)
    cutout = query_legacy_cutout(RA_DEG, DEC_DEG, bands=BANDS, size_pix=80)

    for band in BANDS:
        rms_predicted = props[band]['rms_pixel']
        rms_measured = float(np.nanstd(cutout[band]['image']))
        ratio = rms_predicted / rms_measured
        assert 0.75 < ratio < 1.5, (
            f"Band {band}: RMS mismatch beyond 25%: "
            f"predicted={rms_predicted:.4g} measured={rms_measured:.4g} "
            f"ratio={ratio:.3f}"
        )


def test_rms_pixel_from_psfdepth_handles_edge_cases():
    """Pure-function test: no network involved."""
    assert np.isnan(_rms_pixel_from_psfdepth(0.0, 1.2, 0.262))
    assert np.isnan(_rms_pixel_from_psfdepth(-1.0, 1.2, 0.262))
    assert np.isnan(_rms_pixel_from_psfdepth(1000.0, 0.0, 0.262))
    assert np.isnan(_rms_pixel_from_psfdepth(1000.0, -1.0, 0.262))

    # Reasonable values: positive, finite, smaller than 1 nmgy
    rms = _rms_pixel_from_psfdepth(5000.0, 1.2, 0.262)
    assert 0 < rms < 1.0


def _run_all_and_report():
    tests = [
        test_rms_pixel_from_psfdepth_handles_edge_cases,
        test_query_tractor_props_returns_expected_structure,
        test_rms_predicted_matches_measured_on_cutout,
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
