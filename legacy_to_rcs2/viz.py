"""Visualization helpers for the Legacy -> RCS2 degradation (diagnostics/demos).

Kept separate from the pipeline so that matplotlib stays an optional, demo-only
dependency; the import is done lazily inside the function.
"""

import numpy as np


def plot_original_vs_degraded(original_images, degraded_images, bands,
                              out_path, title=None,
                              left_label="Legacy original",
                              right_label="Degraded RCS2"):
    """Save a per-band comparison figure: original Legacy vs degraded RCS2.

    Rows are bands; the two columns are the original Legacy cutout and the
    degraded RCS2 stamp. Each panel is shown with an asinh stretch over a
    robust percentile range so faint structure is visible. The images are NOT
    pixel-aligned -- this is a qualitative side-by-side. Pass FOV-matched
    arrays (e.g. the Legacy cutout centre-cropped to the RCS2 stamp footprint)
    if you want the two columns to cover the same sky area.

    :param original_images: list of 2D arrays, the Legacy cutouts (one per band)
    :param degraded_images: list of 2D arrays, the degraded RCS2 stamps
    :param bands: str or sequence of band letters, same order as the images
    :param out_path: str, path to write the figure (extension sets the format)
    :param title: optional str, figure suptitle
    :param left_label: str, column title for the original images
    :param right_label: str, column title for the degraded images
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

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
    columns = ((0, left_label, original_images), (1, right_label, degraded_images))
    for row, band in enumerate(bands):
        for col, label, images in columns:
            ax = axes[row][col]
            # Drop non-finite pixels (Legacy cutouts can have NaN at edges)
            # so the percentile stretch is well defined.
            img = np.asarray(images[row], dtype=float)
            img = np.where(np.isfinite(img), img, 0.0)
            norm = simple_norm(img, stretch='asinh',
                               min_percent=1.0, max_percent=99.5)
            ax.imshow(img, origin='lower', cmap='gray', norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{band} band", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path
