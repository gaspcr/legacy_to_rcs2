"""Visualization helpers for the Legacy -> RCS2 degradation (diagnostics/demos).

Kept separate from the pipeline so that matplotlib stays an optional, demo-only
dependency; the import is done lazily inside the function.
"""

import numpy as np

# RGB composite tuning: per-band scaling percentile and LogStretch parameter.
PCTL = 99.5
LOG_A = 1000.0


def _rgb_stamp(images, bands, pctl=PCTL, a=LOG_A):
    """Build an 8-bit RGB composite from g, r, z planes (z->R, r->G, g->B).

    Each band is scaled to its own ``pctl`` percentile and then mapped through
    a LogStretch over [0, 1]. Per-band scaling balances the channels regardless
    of their absolute flux units (so it works for both the Legacy cutout and
    the degraded RCS2 stamp).
    """
    from astropy.visualization import make_rgb, ManualInterval, LogStretch

    idx = {b: i for i, b in enumerate(bands)}

    def _scaled(band):
        stamp = np.asarray(images[idx[band]], dtype=float)
        stamp = np.where(np.isfinite(stamp), stamp, 0.0)
        vmax = np.percentile(stamp, pctl)
        return stamp / max(vmax, 1e-12)

    return make_rgb(_scaled('z'), _scaled('r'), _scaled('g'),
                    interval=ManualInterval(vmin=0, vmax=1),
                    stretch=LogStretch(a=a))


def plot_original_vs_degraded(original_images, degraded_images, bands,
                              out_path, title=None,
                              left_label="Legacy original",
                              right_label="Degraded RCS2", rgb=True):
    """Save a comparison figure: original Legacy vs degraded RCS2.

    The two columns are the original Legacy cutout and the degraded RCS2 stamp.
    When ``rgb`` is set and the bands include g, r and z, the top row is a grz
    RGB composite (z->R, r->G, g->B); the remaining rows are the individual
    bands with an asinh stretch over a robust percentile range. The images are
    NOT pixel-aligned -- pass FOV-matched arrays for a same-sky comparison.

    :param original_images: list of 2D arrays, the Legacy cutouts (one per band)
    :param degraded_images: list of 2D arrays, the degraded RCS2 stamps
    :param bands: str or sequence of band letters, same order as the images
    :param out_path: str, path to write the figure (extension sets the format)
    :param title: optional str, figure suptitle
    :param left_label: str, column title for the original images
    :param right_label: str, column title for the degraded images
    :param rgb: bool, add a grz RGB composite row when g, r, z are all present
    :return: out_path
    """
    import matplotlib
    matplotlib.use('Agg')  # headless: no display needed
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    bands = list(bands)
    n = len(bands)
    if not (len(original_images) == len(degraded_images) == n):
        raise ValueError(
            f"bands ({n}), original ({len(original_images)}) and degraded "
            f"({len(degraded_images)}) must have the same length."
        )

    add_rgb = rgb and {'g', 'r', 'z'}.issubset(set(bands))
    columns = ((0, left_label, original_images), (1, right_label, degraded_images))
    nrows = n + (1 if add_rgb else 0)
    fig, axes = plt.subplots(nrows, 2, figsize=(6, 3 * nrows), squeeze=False)

    # Optional RGB composite row at the top.
    if add_rgb:
        for col, label, images in columns:
            ax = axes[0][col]
            ax.imshow(_rgb_stamp(images, bands), origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=10)
        axes[0][0].set_ylabel("RGB (grz)", fontsize=11)

    # Per-band grayscale rows.
    row0 = 1 if add_rgb else 0
    for r, band in enumerate(bands):
        rr = row0 + r
        for col, label, images in columns:
            ax = axes[rr][col]
            # Drop non-finite pixels (Legacy cutouts can have NaN at edges)
            # so the percentile stretch is well defined.
            img = np.asarray(images[r], dtype=float)
            img = np.where(np.isfinite(img), img, 0.0)
            norm = simple_norm(img, stretch='asinh',
                               min_percent=1.0, max_percent=99.5)
            ax.imshow(img, origin='lower', cmap='gray', norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if rr == 0:  # no RGB row -> column titles go on the first band row
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{band} band", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path
