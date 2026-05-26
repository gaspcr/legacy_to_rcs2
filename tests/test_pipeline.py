"""Wiring tests for legacy_to_rcs2.pipeline (Legacy DR10 -> RCS2).

These tests verify the orchestration *wiring* without hitting the network
or running the (reproject-dependent) degradation core: the network queries
and the degradation entry point are monkeypatched with recording stubs, so
we can assert exactly what arguments the pipeline forwards. The one test
that runs the real degradation is skipped automatically if `reproject`
is not installed.

Run as:
    python tests/test_pipeline.py
"""

import math
import os
import pathlib
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import legacy_to_rcs2.pipeline as pipeline


# --- fixtures -------------------------------------------------------------

# RCS2 conditions per band (median=0 is option A). Distinct per band so we
# can assert the right band's draw reaches the right degradation call.
RCS2_STATS = {
    'g': dict(median=0.0, rms=19.0, seeing=0.84, exp_time=240.0,
              zero_point=26.43, gain=1.57),
    'r': dict(median=0.0, rms=31.0, seeing=0.80, exp_time=480.0,
              zero_point=25.95, gain=1.61),
    'z': dict(median=0.0, rms=43.0, seeing=0.69, exp_time=360.0,
              zero_point=24.80, gain=1.55),
}

# Legacy tractor props per band. FWHM is deliberately *broader* than the
# RCS2 seeing above, mirroring the real regime (DECam ~1.1-1.3", RCS2 ~0.7-0.8").
TRACTOR_PROPS = {
    'g': dict(fwhm_arcsec=1.30, rms_pixel=0.011),
    'r': dict(fwhm_arcsec=1.10, rms_pixel=0.022),
    'z': dict(fwhm_arcsec=1.05, rms_pixel=0.033),
}


class StubBandSampler:
    """Minimal sampler matching the BandSampler contract used by the pipeline."""
    def __init__(self, band):
        self.band = band

    def sample(self):
        return dict(RCS2_STATS[self.band])


def make_sampler(bands='grz'):
    return {b: StubBandSampler(b) for b in bands}


def make_images(bands='grz', size=80):
    """One flat-noise Legacy image per band, value encodes the band index."""
    return [np.full((size, size), float(i), dtype=np.float32)
            for i, _ in enumerate(bands)]


def make_cutout(bands='grz', size=80):
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    return {b: {'image': np.full((size, size), float(i), dtype=np.float32),
                'hdr': hdr.copy()}
            for i, b in enumerate(bands)}


class _Patcher:
    """Save/restore attributes on the pipeline module."""
    def __init__(self):
        self._saved = {}

    def set(self, name, value):
        if name not in self._saved:
            self._saved[name] = getattr(pipeline, name)
        setattr(pipeline, name, value)

    def restore(self):
        for name, value in self._saved.items():
            setattr(pipeline, name, value)
        self._saved.clear()


def _record_degrade_calls():
    """Return (spy, calls): spy mimics legacy_to_rcs2, records its kwargs."""
    calls = []

    def spy(image, band, **kwargs):
        calls.append((band, dict(kwargs)))
        out = kwargs.get('out_size', pipeline.RCS2_STAMP_PIX)
        return np.zeros((out, out), dtype=np.float32)

    return spy, calls


def _passthrough_zero_point(mag_change=0.0):
    """Spy for zero_point_change: records args, returns images unscaled."""
    rec = {}

    def spy(images, original_zp, target_zp, target_rms, rms_frac_thresh=0.1):
        rec['original_zp'] = original_zp
        rec['target_zp'] = list(target_zp)
        rec['target_rms'] = list(target_rms)
        rec['rms_frac_thresh'] = rms_frac_thresh
        return images, mag_change

    return spy, rec


# --- tests ----------------------------------------------------------------

def test_degrade_images_wiring():
    """Each band's degradation call carries the correct RCS2/Legacy values."""
    bands = 'grz'
    p = _Patcher()
    spy, calls = _record_degrade_calls()
    zp_spy, zp_rec = _passthrough_zero_point()
    p.set('legacy_to_rcs2', spy)
    p.set('zero_point_change', zp_spy)
    try:
        ok, degraded, mag_change = pipeline.degrade_images(
            make_images(bands), TRACTOR_PROPS, make_sampler(bands),
            bands=bands, log_file=os.devnull,
        )
    finally:
        p.restore()

    assert ok is True
    assert len(degraded) == len(bands)
    assert mag_change == 0.0
    assert [c[0] for c in calls] == list(bands), "bands degraded out of order"

    for band, kw in calls:
        s = RCS2_STATS[band]
        # RCS2 single exposures: NO coadd->single zero-point correction.
        assert kw['lsst_single_exposure_zero_point'] is None
        # MegaCam gain and conditions come from the sampler.
        assert kw['lsst_ccd_gain'] == s['gain']
        assert kw['exp_time'] == s['exp_time']
        assert kw['lsst_zero_point'] == s['zero_point']
        assert kw['lsst_fwhm'] == s['seeing']
        assert kw['background_noise'] == s['rms']
        assert kw['background_median'] == s['median']
        # Legacy PSF: Gaussian of the tractor FWHM, no 2D PSF model.
        assert kw['hsc_fwhm'] == TRACTOR_PROPS[band]['fwhm_arcsec']
        assert kw['hsc_psf'] is None
        # Output geometry: RCS2 pixel scale and 65 px stamp.
        assert kw['target_pix_scale'] == pipeline.RCS2_PIX_SCALE
        assert kw['hsc_pix_scale'] == pipeline.LEGACY_PIX_SCALE
        assert kw['out_size'] == pipeline.RCS2_STAMP_PIX


def test_zero_point_step_uses_legacy_zp():
    """The zero-point step starts from the Legacy nanomaggie ZP (22.5)."""
    bands = 'grz'
    p = _Patcher()
    spy, _ = _record_degrade_calls()
    zp_spy, zp_rec = _passthrough_zero_point()
    p.set('legacy_to_rcs2', spy)
    p.set('zero_point_change', zp_spy)
    try:
        pipeline.degrade_images(
            make_images(bands), TRACTOR_PROPS, make_sampler(bands),
            bands=bands, zp_rms_frac_thresh=0.3, log_file=os.devnull,
        )
    finally:
        p.restore()

    assert zp_rec['original_zp'] == pipeline.LEGACY_ZERO_POINT == 22.5
    assert zp_rec['target_zp'] == [RCS2_STATS[b]['zero_point'] for b in bands]
    assert zp_rec['target_rms'] == [RCS2_STATS[b]['rms'] for b in bands]
    assert zp_rec['rms_frac_thresh'] == 0.3


def test_band_count_not_hardcoded():
    """Works for any number of bands (regression: the fork hardcoded range(5))."""
    for bands in ('grz', 'gr', 'g'):
        p = _Patcher()
        spy, calls = _record_degrade_calls()
        zp_spy, _ = _passthrough_zero_point()
        p.set('legacy_to_rcs2', spy)
        p.set('zero_point_change', zp_spy)
        try:
            ok, degraded, _ = pipeline.degrade_images(
                make_images(bands), TRACTOR_PROPS, make_sampler(bands),
                bands=bands, log_file=os.devnull,
            )
        finally:
            p.restore()
        assert ok is True
        assert len(degraded) == len(bands)
        assert len(calls) == len(bands)


def test_zero_point_failure_returns_false():
    """If the zero-point change exceeds the max magnitude change, fail cleanly."""
    bands = 'grz'
    p = _Patcher()
    spy, calls = _record_degrade_calls()
    zp_spy, _ = _passthrough_zero_point(mag_change=5.0)  # > default max 2
    p.set('legacy_to_rcs2', spy)
    p.set('zero_point_change', zp_spy)
    try:
        ok, degraded, mag_change = pipeline.degrade_images(
            make_images(bands), TRACTOR_PROPS, make_sampler(bands),
            bands=bands, zp_max_mag_change=2, log_file=os.devnull,
        )
    finally:
        p.restore()
    assert ok is False
    assert degraded is None
    assert mag_change == 0.0
    assert len(calls) == 0, "should not degrade after a failed zero-point step"


def test_query_and_degrade_forwards_query_params():
    """query_and_degrade calls the Legacy queries with the right params and degrades."""
    bands = 'grz'
    p = _Patcher()
    q_calls = {}

    def fake_cutout(ra, dec, bands='grz', layer='ls-dr10', pixscale=None,
                    size_pix=None, timeout=None, **kw):
        q_calls['cutout'] = dict(ra=ra, dec=dec, bands=bands, layer=layer,
                                 pixscale=pixscale, size_pix=size_pix)
        return make_cutout(bands)

    def fake_tractor(ra, dec, bands='grz', search_radius_arcsec=None,
                     pixscale=None, timeout=None, **kw):
        q_calls['tractor'] = dict(ra=ra, dec=dec, bands=bands,
                                  pixscale=pixscale)
        return TRACTOR_PROPS

    spy, calls = _record_degrade_calls()
    zp_spy, _ = _passthrough_zero_point()
    p.set('query_legacy_cutout', fake_cutout)
    p.set('query_tractor_props', fake_tractor)
    p.set('legacy_to_rcs2', spy)
    p.set('zero_point_change', zp_spy)
    try:
        ok, degraded, _ = pipeline.query_and_degrade(
            150.0, 2.0, make_sampler(bands), bands=bands, log_file=os.devnull,
        )
    finally:
        p.restore()

    assert ok is True
    assert len(degraded) == len(bands)
    assert q_calls['cutout']['bands'] == bands
    assert q_calls['cutout']['layer'] == 'ls-dr10'
    assert q_calls['cutout']['pixscale'] == pipeline.LEGACY_PIX_SCALE
    assert q_calls['tractor']['bands'] == bands
    assert q_calls['tractor']['pixscale'] == pipeline.LEGACY_PIX_SCALE


def test_query_failure_returns_false():
    """A failing query is caught, logged, and reported as failure (not raised)."""
    bands = 'grz'
    p = _Patcher()

    def boom(*a, **k):
        raise RuntimeError("network down")

    p.set('query_legacy_cutout', boom)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, 'err.log')
            ok, degraded, mag_change = pipeline.query_and_degrade(
                150.0, 2.0, make_sampler(bands), bands=bands, log_file=log,
            )
            assert os.path.exists(log)
            with open(log) as fh:
                assert 'network down' in fh.read()
    finally:
        p.restore()
    assert ok is False
    assert degraded is None
    assert mag_change == 0.0


def test_cutout_cache_roundtrip():
    """write_cutout_cache -> read_cutout_cache preserves images and props."""
    bands = 'grz'
    cutout = make_cutout(bands)
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'src_000001')
        pipeline.write_cutout_cache(base, cutout, TRACTOR_PROPS,
                                    legacy_pix_scale=0.262, bands=bands)
        images, props, pix_scale = pipeline.read_cutout_cache(base, bands)

    assert pix_scale == 0.262
    assert len(images) == len(bands)
    for i, band in enumerate(bands):
        assert np.allclose(images[i], cutout[band]['image'])
        assert props[band]['fwhm_arcsec'] == TRACTOR_PROPS[band]['fwhm_arcsec']
        assert math.isclose(props[band]['rms_pixel'],
                            TRACTOR_PROPS[band]['rms_pixel'], rel_tol=1e-6)


def test_read_and_degrade_uses_cached_props():
    """read_and_degrade reads the cache and forwards the cached FWHM/pixscale."""
    bands = 'grz'
    cutout = make_cutout(bands)
    p = _Patcher()
    spy, calls = _record_degrade_calls()
    zp_spy, _ = _passthrough_zero_point()
    p.set('legacy_to_rcs2', spy)
    p.set('zero_point_change', zp_spy)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'src_000002')
            pipeline.write_cutout_cache(base, cutout, TRACTOR_PROPS,
                                        legacy_pix_scale=0.262, bands=bands)
            ok, degraded, _ = pipeline.read_and_degrade(
                base, make_sampler(bands), bands=bands, log_file=os.devnull,
            )
    finally:
        p.restore()

    assert ok is True
    assert len(degraded) == len(bands)
    for band, kw in calls:
        assert kw['hsc_fwhm'] == TRACTOR_PROPS[band]['fwhm_arcsec']
        assert kw['hsc_pix_scale'] == 0.262


def test_degrade_real_if_reproject_available():
    """Smoke test of the real degradation core. Skipped without reproject."""
    try:
        import reproject  # noqa: F401
    except ImportError:
        print("    SKIP: reproject not installed (real degradation not exercised)")
        return

    bands = 'grz'
    rng = np.random.default_rng(0)
    # Faint flat field (nanomaggies) so the Legacy noise is well below the
    # RCS2 target rms and add_noise does not raise.
    images = [rng.normal(0.0, 1e-3, size=(80, 80)).astype(np.float32)
              for _ in bands]
    # Sharper Legacy PSF than the RCS2 seeing so the PSF convolution runs.
    sharp_props = {b: dict(fwhm_arcsec=0.5, rms_pixel=1e-3) for b in bands}

    ok, degraded, _ = pipeline.degrade_images(
        images, sharp_props, make_sampler(bands), bands=bands,
        log_file=os.devnull,
    )
    assert ok is True, "real degradation failed"
    for img in degraded:
        assert img.shape == (pipeline.RCS2_STAMP_PIX, pipeline.RCS2_STAMP_PIX)
        assert np.all(np.isfinite(img))


def _run_all_and_report():
    tests = [
        test_degrade_images_wiring,
        test_zero_point_step_uses_legacy_zp,
        test_band_count_not_hardcoded,
        test_zero_point_failure_returns_false,
        test_query_and_degrade_forwards_query_params,
        test_query_failure_returns_false,
        test_cutout_cache_roundtrip,
        test_read_and_degrade_uses_cached_props,
        test_degrade_real_if_reproject_available,
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
