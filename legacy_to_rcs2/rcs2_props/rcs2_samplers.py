"""RCS2 observing-condition samplers for the Legacy -> RCS2 degradation.

The pipeline draws the RCS2 target conditions, once per band per source, via
    rcs2_sampler[band].sample() -> {median, rms, seeing, exp_time,
                                    zero_point, gain}
where ``rcs2_sampler`` is a dict ``{band: BandSampler}`` built from the
characterize_rcs2 CSV by ``load_rcs2_sampler``.

Method: EMPIRICAL BOOTSTRAP ("option A"). Each ``sample()`` returns a single
*real* characterized RCS2 frame's properties drawn at random, so the joint
distribution (and all correlations between seeing, rms, zero point, ...) is
reproduced exactly with no distributional assumptions. ``median`` is fixed to
0.0 because the training stamps are sky-subtracted (see the
feedback-rcs2-median-zero note). This is the recommended option in
``docs/degradation_and_sampler_note.md``; the alternative GMM ("option B")
would replace only this module -- the factory contract below is shared.
"""

import os

import numpy as np
import pandas as pd


# Target sky level: option A forces 0 to match sky-subtracted training stamps.
MEDIAN_OPTION_A = 0.0

# CSV columns drawn as a coherent tuple from a single real frame. (The CSV
# also stores 'median' and 'n_stars'; 'median' is intentionally NOT used here.)
SAMPLED_COLUMNS = ('seeing', 'rms', 'exp_time', 'zero_point', 'gain')


class BandSampler:
    """Bootstrap sampler for one band: draws a real frame's properties."""

    def __init__(self, band, table, rng):
        """:param band: str band letter.
        :param table: dict {column: 1D np.ndarray} over SAMPLED_COLUMNS,
            all arrays the same length (one entry per usable RCS2 frame).
        :param rng: numpy.random.Generator used to pick rows.
        """
        self.band = band
        self._table = table
        self._n = len(table['seeing'])
        self._rng = rng

    def __len__(self):
        return self._n

    def reseed(self, rng):
        """Replace the RNG. Call inside each worker process so parallel
        workers do not draw identical bootstrap sequences (see reseed_samplers).

        :param rng: numpy.random.Generator (or a seed accepted by default_rng)
        """
        self._rng = rng if isinstance(rng, np.random.Generator) \
            else np.random.default_rng(rng)

    def sample(self):
        """Draw one real frame's RCS2 conditions.

        :return: dict {median, rms, seeing, exp_time, zero_point, gain};
            median is always 0.0 (option A), the rest are a coherent tuple
            from a single characterized frame.
        """
        i = int(self._rng.integers(self._n))
        return {
            'median': MEDIAN_OPTION_A,
            'rms': float(self._table['rms'][i]),
            'seeing': float(self._table['seeing'][i]),
            'exp_time': float(self._table['exp_time'][i]),
            'zero_point': float(self._table['zero_point'][i]),
            'gain': float(self._table['gain'][i]),
        }


def load_rcs2_sampler(rcs2_props_csv, bands='grz', seed=None):
    """Build the per-band empirical bootstrap sampler from a characterize_rcs2 CSV.

    :param rcs2_props_csv: str, path to the CSV produced by
        legacy_to_rcs2.rcs2_props.characterize_rcs2 (one row per
        characterized RCS2 frame).
    :param bands: str, concatenated band letters (default 'grz').
    :param seed: optional int seed for reproducibility; each band gets an
        independent RNG spawned from it.
    :return: dict {band: BandSampler}; each BandSampler.sample() returns
        {median, rms, seeing, exp_time, zero_point, gain} with median == 0.0.
    :raises ValueError: if a required column is missing or a requested band
        has no usable frames.
    """
    df = pd.read_csv(rcs2_props_csv)
    required = ('band',) + SAMPLED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{rcs2_props_csv} is missing required column(s) {missing}; "
            f"expected at least {list(required)}."
        )

    # Independent, reproducible RNG per band.
    band_seeds = np.random.SeedSequence(seed).spawn(len(bands))

    samplers = {}
    for band, band_seed in zip(bands, band_seeds):
        # Keep only this band's rows with all sampled properties present
        # (drops e.g. the rare frame whose seeing fit failed -> NaN).
        rows = df[df['band'] == band].dropna(subset=list(SAMPLED_COLUMNS))
        if len(rows) == 0:
            raise ValueError(
                f"No usable RCS2 frames for band {band!r} in {rcs2_props_csv} "
                f"(after dropping rows with missing {list(SAMPLED_COLUMNS)})."
            )
        table = {c: rows[c].to_numpy(dtype=float) for c in SAMPLED_COLUMNS}
        samplers[band] = BandSampler(band, table, np.random.default_rng(band_seed))
    return samplers


def reseed_samplers(samplers, base_seed=None):
    """Give each band sampler a fresh independent RNG.

    Call this once inside each worker process (e.g. in a multiprocessing Pool
    initializer): workers forked from a parent share the parent's RNG state and
    would otherwise draw identical bootstrap sequences. With ``base_seed=None``
    each worker reseeds from OS entropy (independent streams); with an explicit
    ``base_seed`` the streams are mixed with the process id so workers diverge
    while a single-process run stays reproducible.

    :param samplers: dict {band: BandSampler} from load_rcs2_sampler
    :param base_seed: optional int base seed
    :return: None (samplers are reseeded in place)
    """
    if base_seed is None:
        band_seeds = np.random.SeedSequence().spawn(len(samplers))
    else:
        band_seeds = np.random.SeedSequence([base_seed, os.getpid()]).spawn(len(samplers))
    for sampler, band_seed in zip(samplers.values(), band_seeds):
        sampler.reseed(np.random.default_rng(band_seed))
