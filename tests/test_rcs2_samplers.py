"""Tests for legacy_to_rcs2.rcs2_props.rcs2_samplers (empirical bootstrap).

The core tests build a sampler from a small synthetic CSV (self-contained, no
external data). One smoke test runs against the real characterize_rcs2 CSV and
is skipped if that file is not present.

Run as:
    python tests/test_rcs2_samplers.py
"""

import math
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from legacy_to_rcs2.rcs2_props.rcs2_samplers import (
    load_rcs2_sampler,
    reseed_samplers,
    MEDIAN_OPTION_A,
    SAMPLED_COLUMNS,
)

REAL_CSV = '/data/estudiantes/riugarte/rcs2/rcs2_props_sample/rcs2_props.csv'
CONTRACT_KEYS = {'median', 'rms', 'seeing', 'exp_time', 'zero_point', 'gain'}


def _write_synthetic_csv(path, include_nan_seeing=True):
    """A tiny but realistic characterize_rcs2-style CSV, 3 g + 3 r rows.

    One g row has NaN seeing (a failed seeing fit) that must be dropped.
    """
    rows = [
        # band, frame_id, exp_time, gain, zero_point, seeing, rms, median, n_stars
        ('g', 'g1', 240., 1.55, 26.40, 0.80, 18.0, 500., 30),
        ('g', 'g2', 240., 1.58, 26.45, 0.90, 20.0, 510., 25),
        ('g', 'g3', 240., 1.60, 26.43, 0.70, 19.0, 505., 40),
        ('r', 'r1', 480., 1.61, 25.95, 0.75, 30.0, 1380., 35),
        ('r', 'r2', 480., 1.62, 25.96, 0.82, 32.0, 1390., 28),
        ('r', 'r3', 480., 1.60, 25.94, 0.79, 31.0, 1385., 33),
    ]
    if include_nan_seeing:
        rows.append(('g', 'g_bad', 240., 1.57, 26.41, float('nan'), 19.5, 508., 3))
    df = pd.DataFrame(rows, columns=['band', 'frame_id', 'exp_time', 'gain',
                                     'zero_point', 'seeing', 'rms', 'median',
                                     'n_stars'])
    df.to_csv(path, index=False)


def test_sample_contract():
    """sample() returns exactly the 6 contract keys, median 0, all finite."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'props.csv')
        _write_synthetic_csv(csv)
        sampler = load_rcs2_sampler(csv, bands='gr', seed=0)
        assert set(sampler) == set('gr')
        for band in 'gr':
            s = sampler[band].sample()
            assert set(s) == CONTRACT_KEYS, f"{band}: keys {set(s)}"
            assert s['median'] == MEDIAN_OPTION_A == 0.0
            for k, v in s.items():
                assert isinstance(v, float) and math.isfinite(v), f"{band} {k}={v}"


def test_bootstrap_returns_real_rows():
    """Every draw is a coherent tuple from a single real frame."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'props.csv')
        _write_synthetic_csv(csv)
        df = pd.read_csv(csv)
        sampler = load_rcs2_sampler(csv, bands='g', seed=1)

        g_rows = df[(df['band'] == 'g')].dropna(subset=list(SAMPLED_COLUMNS))
        real_tuples = {
            tuple(round(r[c], 6) for c in SAMPLED_COLUMNS)
            for _, r in g_rows.iterrows()
        }
        for _ in range(200):
            s = sampler['g'].sample()
            drawn = tuple(round(s[c], 6) for c in SAMPLED_COLUMNS)
            assert drawn in real_tuples, f"{drawn} is not a real frame"


def test_nan_seeing_row_dropped():
    """The NaN-seeing frame is excluded from the pool and never sampled."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'props.csv')
        _write_synthetic_csv(csv, include_nan_seeing=True)
        sampler = load_rcs2_sampler(csv, bands='g', seed=2)
        assert len(sampler['g']) == 3, "expected 3 usable g rows (NaN dropped)"
        for _ in range(100):
            assert math.isfinite(sampler['g'].sample()['seeing'])


def test_reproducible_with_seed():
    """Same seed -> identical sequence; different seed -> different sequence."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'props.csv')
        _write_synthetic_csv(csv)

        def draw(seed, n=50):
            s = load_rcs2_sampler(csv, bands='g', seed=seed)
            return [s['g'].sample()['seeing'] for _ in range(n)]

        assert draw(123) == draw(123), "same seed must reproduce"
        assert draw(123) != draw(456), "different seeds should differ"


def test_reseed_changes_stream():
    """reseed_samplers replaces the RNG and changes subsequent draws."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'props.csv')
        _write_synthetic_csv(csv)
        sampler = load_rcs2_sampler(csv, bands='g', seed=7)
        before = [sampler['g'].sample()['seeing'] for _ in range(50)]
        reseed_samplers(sampler, base_seed=999)
        after = [sampler['g'].sample()['seeing'] for _ in range(50)]
        assert before != after, "reseed should change the draw sequence"


def test_missing_band_raises():
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'props.csv')
        _write_synthetic_csv(csv)
        try:
            load_rcs2_sampler(csv, bands='z')  # no z rows
            assert False, "expected ValueError for band with no rows"
        except ValueError as e:
            assert 'z' in str(e)


def test_missing_column_raises():
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, 'bad.csv')
        pd.DataFrame({'band': ['g'], 'seeing': [0.8]}).to_csv(csv, index=False)
        try:
            load_rcs2_sampler(csv, bands='g')
            assert False, "expected ValueError for missing columns"
        except ValueError as e:
            assert 'missing' in str(e).lower()


def test_real_csv_smoke():
    """Smoke test on the real characterize_rcs2 CSV (skipped if absent)."""
    if not os.path.exists(REAL_CSV):
        print(f"    SKIP: {REAL_CSV} not found")
        return
    sampler = load_rcs2_sampler(REAL_CSV, bands='grz', seed=42)
    assert set(sampler) == set('grz')
    for band in 'grz':
        assert len(sampler[band]) > 50, f"{band}: too few frames"
        for _ in range(200):
            s = sampler[band].sample()
            assert s['median'] == 0.0
            assert 0.4 < s['seeing'] < 2.0, f"{band} seeing {s['seeing']}"
            assert s['rms'] > 0
            assert 1.0 < s['gain'] < 2.0
            assert 20 < s['zero_point'] < 30
            assert s['exp_time'] > 0


def _run_all_and_report():
    tests = [
        test_sample_contract,
        test_bootstrap_returns_real_rows,
        test_nan_seeing_row_dropped,
        test_reproducible_with_seed,
        test_reseed_changes_stream,
        test_missing_band_raises,
        test_missing_column_raises,
        test_real_csv_smoke,
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
