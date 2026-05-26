"""Tests for legacy_to_rcs2.viz (offline, synthetic arrays)."""

import os
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from legacy_to_rcs2.viz import plot_original_vs_degraded


def test_plot_creates_nonempty_png():
    """A comparison figure (with the grz RGB row) is written and non-empty."""
    rng = np.random.default_rng(0)
    original = [rng.normal(0, 1, (46, 46)) for _ in 'grz']
    degraded = [rng.normal(0, 1, (65, 65)) for _ in 'grz']
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'cmp.png')
        ret = plot_original_vs_degraded(original, degraded, 'grz', out,
                                        title='demo')
        assert ret == out
        assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_rgb_toggle():
    """The grz RGB row is optional; both with and without it render."""
    rng = np.random.default_rng(1)
    original = [rng.normal(5, 1, (46, 46)) for _ in 'grz']
    degraded = [rng.normal(5, 1, (65, 65)) for _ in 'grz']
    with tempfile.TemporaryDirectory() as tmp:
        with_rgb = plot_original_vs_degraded(original, degraded, 'grz',
                                             os.path.join(tmp, 'rgb.png'), rgb=True)
        no_rgb = plot_original_vs_degraded(original, degraded, 'grz',
                                           os.path.join(tmp, 'norgb.png'), rgb=False)
        assert os.path.getsize(with_rgb) > 0 and os.path.getsize(no_rgb) > 0


def test_plot_handles_nan_pixels():
    """Non-finite pixels in the input must not break the percentile stretch."""
    original = [np.full((46, 46), np.nan) for _ in 'g']
    original[0][20:25, 20:25] = 1.0  # a few finite pixels
    degraded = [np.ones((65, 65)) for _ in 'g']
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'cmp.png')
        plot_original_vs_degraded(original, degraded, 'g', out)
        assert os.path.getsize(out) > 0


def test_plot_length_mismatch_raises():
    original = [np.zeros((10, 10)) for _ in 'gr']
    degraded = [np.zeros((10, 10)) for _ in 'grz']  # mismatched length
    with tempfile.TemporaryDirectory() as tmp:
        try:
            plot_original_vs_degraded(original, degraded, 'grz',
                                      os.path.join(tmp, 'x.png'))
            assert False, "expected ValueError on length mismatch"
        except ValueError as e:
            assert 'same length' in str(e)


def _run_all_and_report():
    tests = [
        test_plot_creates_nonempty_png,
        test_plot_rgb_toggle,
        test_plot_handles_nan_pixels,
        test_plot_length_mismatch_raises,
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
