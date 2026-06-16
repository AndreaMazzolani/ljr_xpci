import numpy as np
from .fourier_tools import gen_fft2

def filter_ft(data, filter_fraction_by_axis, mode="cut"):
    """
    Apply a Fourier-domain filter along selected axes.

    Parameters
    ----------
    data : np.ndarray, shape (N0, N1, ..., Nk)
        Input array to be filtered.

    filter_fraction_by_axis : sequence, length data.ndim
        Filtering fraction for each axis, in the same order as data.shape. Use None for axes that should not be filtered. For example, if data has shape (Ns, Ntheta, Nx), use (None, None, 0.4) to filter only the detector x-axis.

    mode : {"cut", "max"}, optional
        "cut" removes frequencies above fraction*max_frequency along each selected axis. "max" removes Fourier coefficients whose magnitude is below fraction*max(abs(F)).

    Returns
    -------
    filtered_data : np.ndarray, shape (N0, N1, ..., Nk)
        Filtered array with the same shape as data.
    """

    if mode not in ("cut", "max"): raise ValueError('mode must be either "cut" or "max".')

    filtered_data = np.asarray(data); filter_fraction_by_axis = tuple(filter_fraction_by_axis); n_axes = filtered_data.ndim  # input data and axis-wise filter settings
    if len(filter_fraction_by_axis) != n_axes: raise ValueError("filter_fraction_by_axis must have one entry for each data dimension.")

    working_data = np.asarray(data).copy()

    for axis, current_fraction in enumerate(filter_fraction_by_axis):
        if current_fraction is None: continue

        axis_data = np.moveaxis(working_data, axis, -1)
        n_samples = axis_data.shape[-1]; sample_positions = np.arange(0, n_samples, dtype=float)

        data_fft, frequencies = gen_fft2(1, sample_positions, axis_data, 0)
        data_fft[..., np.abs(frequencies) > np.max(frequencies) * float(current_fraction)] = 0

        axis_data_filtered = gen_fft2(-1, frequencies, data_fft, [sample_positions[0], sample_positions[-1]], n_samples)[0]
        working_data = np.moveaxis(axis_data_filtered, -1, axis)
        # print("axis =", axis, "n_samples =", n_samples, "frequencies.shape =", frequencies.shape, "frequencies[:3] =", frequencies[:3], "frequencies[-3:] =", frequencies[-3:])

    return working_data









