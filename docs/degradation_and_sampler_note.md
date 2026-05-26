# Legacy → RCS2 degradation pipeline — design note for the commission

**Sub-project:** `legacy_to_rcs2` (data-augmentation for the RCS2 strong-lens search)
**Status:** characterization stage complete; sampler design under discussion
**Date:** 2026-05-24

> Purpose of this note. (1) Explain in general terms what the pipeline does and
> why. (2) Lay out, with rationale, each design decision taken so far. (3) Focus
> the discussion on one open methodological choice that we consider a key point
> before continuing: **how to sample the target RCS2 observing conditions** —
> an empirical (bootstrap) sampler vs. a Gaussian Mixture Model (GMM), the latter
> being what the upstream code we forked uses.

---

## 1. What we are building, and why

The scientific goal of the thesis/paper is the **automated detection of
galaxy-scale strong gravitational lenses in RCS2** (Red Cluster Survey 2,
CFHT/MegaCam) using a CNN classifier.

The bottleneck is **training data**: RCS2 has very few confirmed/visually
identified lens candidates — too few to train a robust CNN. The standard
remedy is data augmentation with **realistic** positives and negatives.

Our strategy: take **Legacy Survey DR10** images — which are deeper, sharper,
and cover the same `g, r, z` bands we train on — and **degrade them to look
like RCS2 single-exposure MegaCam frames**. The degraded stamps enter the CNN
training set alongside real RCS2 stamps. Because Legacy is higher quality, we
can always degrade *down* to RCS2 conditions in a controlled way; the reverse
would not be possible.

This pipeline is a **fork of `furcelay/HSC_to_LSST`** (F. Urcelay, 2024, MIT),
which degraded HSC images to LSST quality. We kept its survey-agnostic
degradation core and replaced the survey-specific input/output ends.

### Why Legacy DR10 as the source
- It has `g, r, z` — exactly the bands we train on (we **exclude `i`** because
  RCS2 `i`-band is distorted in many fields).
- It is deeper and has better seeing than RCS2, so degradation is well-posed.
- It has a public cutout service and a calibrated source catalogue
  (`ls_dr10.tractor`) that gives us the per-band PSF and noise of each input
  cutout, which the degradation needs.

---

## 2. Pipeline architecture (data flow)

For each target position `(ra, dec)`:

1. **Download** a Legacy DR10 `grz` cutout at the **native pixel scale
   0.262″/px** (no server-side resampling — we control the resampling
   ourselves to conserve flux).
2. **Characterize the input** by querying `ls_dr10.tractor` for the nearest
   source: derive the per-band PSF **FWHM** (from `psfsize`) and per-pixel
   **RMS** (from `psfdepth`, via the Gaussian effective area). This tells the
   degradation what it is starting from.
3. **Sample the target RCS2 conditions** — `(median, rms, seeing, exp_time,
   zero_point, gain)` per band — from a model built on **real RCS2 frames**.
   **← This note is about how to build that model (Section 4).**
4. **Degrade** Legacy → RCS2:
   - rescale to the RCS2 **zero point**;
   - convolve with a Gaussian PSF to match RCS2 **seeing**;
   - **reproject** 0.262″ → **0.185″/px** (MegaCam) conserving flux;
   - add **Poisson + sky noise** to match the RCS2 background;
   - crop to **65 × 65 px (= 12″)**, the CNN stamp size.

The output stamps should be statistically indistinguishable, in their
observing conditions, from the real RCS2 stamps the CNN sees in production.

---

## 3. Design decisions taken so far (audit trail)

Each decision is flagged as **scientific** (affects the data/statistics) or
**engineering** (affects correctness/efficiency only).

| # | Decision | Type | Rationale |
|---|----------|------|-----------|
| D1 | Use bands `g, r, z`; **exclude `i`** | scientific | RCS2 `i`-band is distorted in many fields; including it would inject artefacts into training. |
| D2 | Source = **Legacy DR10**; degrade *down* to RCS2 | scientific | Legacy is deeper/sharper and shares `grz`; degradation is well-posed only in this direction. |
| D3 | Download cutouts at **native 0.262″/px**, no server resample | engineering | Keep resampling under our control for flux conservation and reproducibility. |
| D4 | Input PSF/noise from **`ls_dr10.tractor`** (`psfsize`,`psfdepth`) | scientific | Per-object, calibrated; converts to FWHM and per-pixel RMS analytically. Validated to ~10 % (g/r), ~20 % (z) vs. measured cutout std. |
| D5 | Target stamp **65 px = 12″** at **0.185″/px** | scientific | Matches the CNN input and MegaCam pixel scale. |
| D6 | **Quality cuts** for which RCS2 frames define "RCS2 conditions": `PHOT_C` present, `CERROR < 1″`, `NASTRO > 20`, `AIRMASS < 1.5` | scientific | Excludes non-photometric / poorly-astrometered / high-airmass frames that would bias the target distribution toward *worse* conditions than production. 73 % of frames pass. |
| D7 | **Discard** frames without `PHOT_C` rather than imputing a band-average ZP | scientific | Imputing would inject a systematic zero-point error; better to drop. |
| D8 | Target **`median` (sky level) fixed at 0** ("option A") | scientific | The training cubes are **sky-subtracted at the stamp level**, and Legacy cutouts are also ≈0-median; forcing the target median to 0 avoids a systematic offset between synthetic and real stamps. The measured per-frame median is still stored for diagnostics and a possible future "option B". |
| D9 | Measure sky `median`/`rms` on a **central 1500×1500 crop**, with robust initial values pre-computed (`sigma_clipped_stats`) before iterative source-masking | engineering | Avoids CCD edge artefacts; the pre-seeding is required because RCS2 Elixir frames are **not** sky-subtracted (sky ≈ 2000 ADU) — without it the estimator returns garbage. |
| D10 | Measure **seeing** by detecting bright unsaturated stars (`DAOStarFinder`) and fitting a circular `Gaussian2D`, taking the median FWHM | scientific | RCS2 headers carry **no SEEING keyword**; this recovers it directly from the image. NaN if < 5 usable stars. |
| D11 | Per-reason **discard logging**; never let one bad frame abort a run | engineering | Auditability of large characterization runs; e.g. distinguishing "no PHOT_C" from "empty frame". |

**Known, non-blocking issue.** The frame-selection script has an off-by-one in
how it numbers extensions for `funpack`, so the `_00` extraction yields an empty
HDU. It cost us 6 of 300 frames (now auto-discarded). It does **not** bias the
property distribution (the remaining frames are real, self-consistent CCDs, and
we sample conditions, not specific CCDs), so we treat it as cleanup.

---

## 4. The open question: how to sample RCS2 observing conditions

### 4.1 The sampler's role and contract

At step 3 the pipeline calls, once per band per target:

```
sample() -> {median, rms, seeing, exp_time, zero_point, gain}
```

These six numbers fully specify the RCS2 conditions to imprint on one degraded
stamp. The question is **what statistical model produces them**. We have
characterized the real RCS2 conditions (Section 5) and must decide how to draw
from that characterization.

### 4.2 How the forked code does it — a Gaussian Mixture Model

The upstream `HSC_to_LSST` ships a **pre-trained GMM per band**
(`sklearn.mixture.BayesianGaussianMixture`, **4 components, full covariance**,
4 features), built on LSST DP0 data.

A GMM is an unsupervised ML model that approximates a probability density as a
weighted sum of *K* multivariate Gaussians:

  p(x) = Σ_k π_k · N(x | μ_k, Σ_k),   with Σ_k π_k = 1,

where **x** is the joint vector of properties `(median, rms, seeing, …)`,
π_k the mixing weights, μ_k the means, and **Σ_k the full covariance matrices**.
The parameters are *fit* to the data (classically by Expectation–Maximization;
the Bayesian variant used here uses variational inference with a prior that can
prune unneeded components, so one sets an upper bound on *K* instead of choosing
it by hand).

**Sampling** = pick a component *k* with probability π_k, then draw
**x ~ N(μ_k, Σ_k)**. Because each component carries a *full* covariance, the
drawn vector **preserves the correlations** between properties (e.g. seeing↔rms).
The upstream `sample()` then floors `rms ≥ 1e-6` (a Gaussian can go negative),
fills in fixed constants (`exp_time` or `zero_point`), and rescales for coadd
depth.

**Why this suited DP0.** DP0 is an LSST *simulation* with many visits/coadds and
continuous coverage of conditions; the team also needed to model both single-
visit and multi-year-coadd regimes and scale analytically between them. A smooth
generative density fit on abundant data is a natural fit there.

### 4.3 Option A — empirical (bootstrap) sampler

`sample()` draws a **real row** from the characterized table at random and
returns its `(seeing, rms, exp_time, zero_point, gain)` as a coherent set,
with `median = 0` (D8).

- **+** Reproduces the **real joint distribution** exactly, correlations included,
  with **no distributional assumptions**.
- **+** ~10 lines, no extra dependency, **immediately auditable** by an astronomer
  ("we replay observed RCS2 frame conditions").
- **+** Robust with our sample size (~100 frames/band).
- **−** **Discrete**: can only emit the observed combinations (no interpolation).

### 4.4 Option B — GMM sampler (as upstream)

Fit a `BayesianGaussianMixture` per band on our table and sample from it.

- **+** **Smooth/continuous**; can interpolate between observed frames and emit
  novel-but-plausible combinations.
- **+** Single framework that could later **scale** across conditions.
- **−** With ~100 frames/band, a 4-component **full-covariance** fit (≈10 covariance
  parameters × components) risks over-fitting / effectively memorizing the data.
- **−** Adds an `sklearn` dependency, a fit/validation step, and a pickled artefact;
  **less directly auditable**.
- **−** Can sample **unphysical tails** (negative rms/seeing) → needs flooring/clipping.

### 4.5 The honest framing

This is **not** "ML vs. not-ML". A full-covariance GMM is a **smoothed,
generative version of the empirical distribution**: with many tight components it
approaches the bootstrap; with few components it is a smooth parametric density
that can interpolate. **Both preserve correlations** (the GMM through Σ_k, the
bootstrap by construction). The real axis of choice is:

> **discrete & exact (bootstrap)** vs. **continuous & smoothed (GMM)** —
> and whether we need to *interpolate/extrapolate* beyond observed RCS2
> conditions, or only to *match* them.

| | DP0 (upstream) | Our RCS2 |
|---|---|---|
| Source | LSST **simulation** | **real** single-exposure frames |
| Sample size | large | ~100 / band |
| Goal | model **and scale** visit↔coadd | **match** one observed regime |
| `exp_time` | varies / time-scaled | ≈ fixed (g 240 s, r 480 s, z 360 s) |
| Interpolation needed? | yes | no |

Two RCS2-specific notes:
- **`gain`** has tiny spread (1.55–1.61); as a GMM dimension it is near-degenerate
  and better treated as a near-constant or carried in the bootstrapped tuple.
  (Upstream had no `gain` dimension at all.)
- With `median = 0` (D8) and `exp_time` essentially fixed per band, only
  `seeing, rms, zero_point` (and weakly `gain`) actually vary — and those come as
  a coherent real tuple in the bootstrap.

### 4.6 Recommendation (for discussion)

We **lean toward Option A (empirical bootstrap)** because the scientific goal is
to **match** observed RCS2 conditions (not to extrapolate), our frames are
**real**, the sample size favours bootstrap robustness, and it is the most
transparent/auditable choice — consistent with our principle of preferring
explicit, physically-legible code. The GMM made sense for DP0's simulated,
abundant, multi-regime data; our situation differs on every one of those axes.

**However**, Option B is fully defensible and we want the commission's view if:
- they prefer a **smooth/continuous** target density;
- they anticipate needing to **extrapolate** to conditions not in our sample;
- they want a **single framework** consistent with the upstream method for
  comparability.

If the commission prefers the GMM, we would fit a `BayesianGaussianMixture` on
the characterized table, **choose K by BIC** (not a fixed 4), exclude/constant
the degenerate `gain` dimension, and **validate** by overplotting sampled vs.
real marginal histograms per band before adopting it.

**Question for the commission:** do we match the observed RCS2 condition
distribution as-is (bootstrap), or model it with a smoothed generative density
(GMM)? This choice sets the statistical character of the entire augmented
training set, which is why we want to settle it before continuing.

---

## 5. Evidence: the characterized RCS2 conditions (294 frames)

Built from `select_rcs2_sample.py` (full archive scan: 986 pointings, 118,738
CCD extensions, 87,198 passing the D6 cuts) → a curated 100-per-band sample →
`characterize_rcs2.py`. Result: **294/300 frames characterized**, 6 discarded as
empty (the off-by-one frames). Per-band medians:

| band | n | seeing″ (median) | seeing range | rms (median, ADU) | sky median (ADU) | zero point (median) | gain (median) | exp_time (s) |
|------|--:|--:|:--:|--:|--:|--:|--:|--:|
| g | 100 | 0.842 | 0.57–1.10 | 19.1 | 505.7 | 26.429 | 1.572 | 240 |
| r | 98 | 0.801 | 0.52–1.47 | 31.4 | 1385.6 | 25.954 | 1.614 | 480 |
| z | 96 | 0.693 | 0.54–0.98 | 42.6 | 2129.5 | 24.795 | 1.550 | 360 |

All trends are physically consistent: seeing improves toward the red
(z < r < g), sky brightness and RMS rise toward the red (no sky subtraction),
and the zero points match the known MegaCam values. Only **1** frame had too
few stars for a seeing fit (NaN). This is the distribution that either sampler
would draw from.

---

*Prepared as a discussion document. Written in English to match the codebase and
paper trajectory; a Spanish version can be produced if the commission prefers.*
