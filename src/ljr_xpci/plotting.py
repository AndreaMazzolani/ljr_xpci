import numpy as np
import matplotlib.pyplot as plt


def contrast_limits(image, saturation_fraction=0.05):
    """
    Compute image contrast limits by excluding the lowest and highest intensity fractions.

    Parameters
    ----------
    image : np.ndarray
        Input image. Singleton dimensions are removed before estimating the contrast limits.

    saturation_fraction : float, optional
        Fraction of pixels excluded at each side of the intensity range.

    Returns
    -------
    lower_limit : float
        Lower display limit.
    upper_limit : float
        Upper display limit.
    """

    image = np.squeeze(np.asarray(image))
    if image.ndim != 2: raise ValueError("image must be 2D after removing singleton dimensions.")
    lower_limit, upper_limit = np.nanquantile(image, [saturation_fraction, 1.0 - saturation_fraction])
    return lower_limit, upper_limit


def show_image(image, z_axis=None, x_axis=None, saturation_fraction=0.05, figure_index=None):
    """
    Display a 2D image with contrast limits based on image quantiles.

    Parameters
    ----------
    image : np.ndarray, shape (Nz, Nx)
        Image to display.

    z_axis : np.ndarray, shape (Nz,), optional
        Vertical-axis coordinates.

    x_axis : np.ndarray, shape (Nx,), optional
        Horizontal-axis coordinates.

    saturation_fraction : float, optional
        Fraction of pixels excluded at each side of the intensity range.

    figure_index : int, optional
        Figure number. If omitted, a new figure is created.

    Returns
    -------
    lower_limit : float
        Lower display limit.
    upper_limit : float
        Upper display limit.
    """

    image = np.squeeze(np.asarray(image))
    if image.ndim != 2: raise ValueError("image must be 2D after removing singleton dimensions.")

    nz, nx = image.shape; z_axis = np.arange(nz) if z_axis is None else np.asarray(z_axis); x_axis = np.arange(nx) if x_axis is None else np.asarray(x_axis)
    if z_axis.size == nx and x_axis.size == nz: image = image.T; nz, nx = image.shape
    if z_axis.size != nz or x_axis.size != nx: raise ValueError("x_axis and z_axis dimensions do not match image.shape.")

    lower_limit, upper_limit = contrast_limits(image, saturation_fraction)
    if figure_index is not None: plt.figure(figure_index); plt.clf()
    else: plt.figure()

    plt.imshow(image, extent=[x_axis[0], x_axis[-1], z_axis[-1], z_axis[0]], aspect="auto", vmin=lower_limit, vmax=upper_limit)
    plt.colorbar()
    return lower_limit, upper_limit