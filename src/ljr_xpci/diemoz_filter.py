import numpy as np
from .fourier_tools import gen_fft2

def apply_single_shot_ei_filter(normalized_projection, gamma_jei, pixel_size, mtf_x=1.0):
    """
    Apply the single-shot edge-illumination retrieval filter in Fourier space.

    The filter is applied along the last axis, which is assumed to be the detector x-axis.

    Parameters
    ----------
    normalized_projection : np.ndarray, shape (..., Nx)
        Flat-field-normalized intensity projection. This is the measured projection in intensity space, before the logarithm is applied. The last dimension contains the detector x samples. Any leading dimensions are treated as independent projections or batch dimensions.

    gamma_jei : float
        Edge-illumination retrieval parameter controlling the strength of the differential phase correction.

    pixel_size : float
        Detector pixel size along the x-axis.

    mtf_x : float or np.ndarray, shape (Nx,), optional
        Detector modulation transfer function along x. Use 1.0 to disable MTF correction.

    Returns
    -------
    retrieved_log_projection : np.ndarray, shape (..., Nx)
        Retrieved projection in log space, computed as -log(abs(filtered_complex_projection)). For a Gaussian-like illumination response, this corresponds to the recovered exponent argument after the Fourier filtering step.

    filtered_complex_projection : np.ndarray, shape (..., Nx)
        Complex filtered intensity projection before taking the absolute value and logarithm.
    """

    normalized_projection = np.asarray(normalized_projection); number_x_pixels = normalized_projection.shape[-1]  # last axis is detector x
    x_positions = pixel_size * np.arange(1, number_x_pixels + 1, dtype=float)  # physical x positions, MATLAB pixel_size.*[1:Nx].'

    projection_fft, spatial_frequencies = gen_fft2(1, x_positions, normalized_projection, 0)
    diemoz_filter = 1.0 / mtf_x / (1.0 - 1j * gamma_jei * spatial_frequencies * 2.0 * np.pi)
    filtered_complex_projection = gen_fft2(-1, spatial_frequencies, projection_fft * diemoz_filter, [x_positions[0], x_positions[-1]], number_x_pixels)[0]
    retrieved_log_projection = -np.log(np.abs(filtered_complex_projection))
    
    return retrieved_log_projection, filtered_complex_projection






