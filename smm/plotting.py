import numpy as np
import matplotlib.pyplot as plt


def dist_colours(indices, cm='hsv', alpha=1.0):
    """
    Takes indices and applies a random colour from the given colour map to
    each subset.

    Parameters
    ----------
    indices : list of slices
        consequtive slices indicating the subsets to colour, sizes sum to N.
    cm : cmap, optional
    alpha : float, optional

    Returns
    -------
    colours : (N, 4) ndarray
    """
    n_samples = sum(ix.stop - ix.start for ix in indices)
    c = np.zeros((n_samples,), dtype=np.int)

    # Number samples as being from simplices
    for i, ix in enumerate(indices):
        c[ix] = i

    # Apply colour to each sample
    cmap = plt.cm.get_cmap(cm, len(indices))
    colours = [cmap(col) for col in c]

    # Update alpha
    for i, col in enumerate(colours):
        colours[i] = (col[0], col[1], col[2], alpha)

    return colours


__all__ = ["dist_colours"]
