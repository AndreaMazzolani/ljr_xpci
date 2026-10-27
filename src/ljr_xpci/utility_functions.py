import numpy as np
from scipy.ndimage import uniform_filter



def err_n(reference_values, comparison_values, axis=None):
    """
    Compute the symmetric relative L2 error between two arrays.

    Parameters
    ----------
    reference_values : np.ndarray
        First input array.
    comparison_values : np.ndarray
        Second input array, same shape as reference_values.
    axis : int or tuple[int], optional
        Axis or axes over which the error is computed. If None, both arrays are flattened.

    Returns
    -------
    relative_error : float or np.ndarray
        Symmetric relative L2 error.
    """

    reference_values = np.asarray(reference_values); comparison_values = np.asarray(comparison_values)
    if reference_values.shape != comparison_values.shape: raise ValueError("The arrays have different dimensions.")

    if axis is None:
        reference_values = reference_values.ravel(); comparison_values = comparison_values.ravel(); numerator = 2.0 * np.linalg.norm(reference_values - comparison_values); denominator = np.linalg.norm(reference_values) + np.linalg.norm(comparison_values)
    else:
        numerator = 2.0 * np.sqrt(np.sum(np.abs(reference_values - comparison_values)**2, axis=axis)); denominator = np.sqrt(np.sum(np.abs(reference_values)**2, axis=axis)) + np.sqrt(np.sum(np.abs(comparison_values)**2, axis=axis))

    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)

def moving_mean_1d(data, window_size, axis):
    data = np.asarray(data, dtype=float); window_size = int(window_size)
    if window_size < 1: raise ValueError("window_size must be positive.")

    moved_data = np.moveaxis(data, axis, 0); n_samples = moved_data.shape[0]
    left_radius = (window_size - 1) // 2; right_radius = window_size // 2
    start_indices = np.maximum(np.arange(n_samples) - left_radius, 0); end_indices = np.minimum(np.arange(n_samples) + right_radius + 1, n_samples)

    cumulative_data = np.concatenate([np.zeros_like(moved_data[:1]), np.cumsum(moved_data, axis=0)], axis=0)
    moving_sum = np.take(cumulative_data, end_indices, axis=0) - np.take(cumulative_data, start_indices, axis=0)
    moving_count = (end_indices - start_indices).reshape((n_samples,) + (1,) * (moved_data.ndim - 1))

    return np.moveaxis(moving_sum / moving_count, 0, axis)

'''
def moving_mean_2d(data, window_size):
    return moving_mean_1d(moving_mean_1d(data, window_size, axis=0), window_size, axis=1)

def moving_mean_3d(data, moving_window_size):
    data = np.asarray(data, dtype=float)
    if data.ndim != 3: raise ValueError("data must have shape (Nz, Ny, Nx).")

    if np.isscalar(moving_window_size):
        window_x = int(moving_window_size); window_y = int(moving_window_size); window_z = data.shape[0]
    else:
        window_x, window_y, window_z = [int(v) for v in moving_window_size]

    if min(window_x, window_y, window_z) < 1: raise ValueError("All moving-window sizes must be positive.")

    return moving_mean_1d(moving_mean_1d(moving_mean_1d(data, window_z, axis=0), window_y, axis=1), window_x, axis=2)
'''

def moving_mean_2d(data, moving_window_size):
    data = np.asarray(data, dtype=float); moving_window_size = int(moving_window_size)
    local_sum = uniform_filter(data, size=(moving_window_size, moving_window_size), mode="constant", cval=0.0) * moving_window_size**2
    local_count = uniform_filter(np.ones_like(data), size=(moving_window_size, moving_window_size), mode="constant", cval=0.0) * moving_window_size**2
    return local_sum / local_count


def moving_mean_3d(data, moving_kernel_size_3d):
    data = np.asarray(data, dtype=float)
    window_x, window_y, window_z = moving_kernel_size_3d
    window_x = int(window_x); window_y = int(window_y); window_z = 1 if window_z is None else int(window_z)
    kernel_size = (window_z, window_y, window_x)

    local_sum = uniform_filter(data, size=kernel_size, mode="constant", cval=0.0) * (window_x * window_y * window_z)
    local_count = uniform_filter(np.ones_like(data), size=kernel_size, mode="constant", cval=0.0) * (window_x * window_y * window_z)

    return local_sum / local_count

def normalise_filter_fraction(filter_fraction_by_axis, ndim):
    if np.isscalar(filter_fraction_by_axis): return tuple(float(filter_fraction_by_axis) for _ in range(ndim))
    filter_fraction_by_axis = tuple(filter_fraction_by_axis)
    if len(filter_fraction_by_axis) != ndim: raise ValueError("filter_fraction_by_axis must be scalar or have one entry for each data dimension.")
    return filter_fraction_by_axis

