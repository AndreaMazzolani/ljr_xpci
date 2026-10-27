import numpy as np
from ljr_xpci.utility_functions import err_n

def normalize_linear_system_determinant(system_matrix, right_hand_side):
    """
    Rescale a linear system so that abs(det(A)) is moved to one using a log-determinant scalar.

    system_matrix : np.ndarray, shape (N, N) or (N, N, Nrow)
        Linear-system matrix or stack of matrices.
    right_hand_side : np.ndarray, shape (N,) or (N, Nrow)
        Right-hand side vector or matrix.
    """

    scaled_matrix = np.asarray(system_matrix, dtype=float).copy(); scaled_rhs = np.asarray(right_hand_side, dtype=float).copy()
    matrix_was_2d = scaled_matrix.ndim == 2; rhs_was_1d = scaled_rhs.ndim == 1

    if matrix_was_2d: scaled_matrix = scaled_matrix[:, :, None]
    elif scaled_matrix.ndim != 3: raise ValueError("system_matrix must have shape (N, N) or (N, N, Nrow).")

    if rhs_was_1d: scaled_rhs = scaled_rhs[:, None]
    elif scaled_rhs.ndim != 2: raise ValueError("right_hand_side must have shape (N,) or (N, Nrow).")

    n_rows, n_cols, n_systems = scaled_matrix.shape
    if n_rows != n_cols: raise ValueError("system_matrix must be square along its first two axes.")
    if scaled_rhs.shape[0] != n_rows: raise ValueError("right_hand_side first dimension must match system_matrix.")

    sign_det, log_abs_det = np.linalg.slogdet(np.moveaxis(scaled_matrix, -1, 0))
    if np.any(sign_det == 0): raise np.linalg.LinAlgError("At least one system matrix is singular.")
    if np.any(~np.isfinite(log_abs_det)): raise np.linalg.LinAlgError("At least one system matrix has non-finite log-determinant.")

    scale_factor = np.exp(-log_abs_det / n_rows)
    scaled_matrix = scaled_matrix * scale_factor[None, None, :]

    if n_systems == 1: scaled_rhs = scaled_rhs * scale_factor[0]
    elif scaled_rhs.shape[1] == n_systems: scaled_rhs = scaled_rhs * scale_factor[None, :]
    else: raise ValueError("right_hand_side must have shape (N,) or (N, Nrow) compatible with system_matrix.")

    if matrix_was_2d: scaled_matrix = scaled_matrix[:, :, 0]
    if rhs_was_1d: scaled_rhs = scaled_rhs[:, 0]

    return scaled_matrix, scaled_rhs, 1

def estimate_material_coefficients(log_projection, thickness, thickness_derivative, j_ei, k0, return_projections=False, fit_weights=None):
    """
    Estimate material coefficients from log projections and material thickness sinograms.

    fit_weights : np.ndarray, shape (Ntheta, Nx), optional
        Ray weights. Larger values give more importance to selected rays during the coefficient fit.
    """

    log_projection = np.asarray(log_projection, dtype=float); thickness = np.asarray(thickness, dtype=float); thickness_derivative = np.asarray(thickness_derivative, dtype=float)
    if thickness.shape != thickness_derivative.shape: raise ValueError("thickness and thickness_derivative must have the same shape.")
    if thickness.ndim != 3: raise ValueError("thickness must have shape (Nv, Ntheta, Nx).")

    number_materials = thickness.shape[0]; sample_shape = thickness.shape[1:]
    if log_projection.shape[-2:] != sample_shape: raise ValueError("The last two dimensions of log_projection must match thickness.shape[1:].")

    log_projection_rows = log_projection.reshape(-1, np.prod(sample_shape))
    number_rows = log_projection_rows.shape[0]; use_refraction = not np.isclose(j_ei, 0.0)
    number_unknowns = 2 * number_materials if use_refraction else number_materials

    thickness_flat = thickness.reshape(number_materials, -1).T
    derivative_flat = thickness_derivative.reshape(number_materials, -1).T

    if fit_weights is None:
        fit_weights_flat = np.ones(thickness_flat.shape[0], dtype=float)
    else:
        fit_weights = np.asarray(fit_weights, dtype=float)
        if fit_weights.shape != sample_shape: raise ValueError("fit_weights must have shape (Ntheta, Nx).")
        fit_weights_flat = fit_weights.ravel()

    thickness_weighted = thickness_flat * fit_weights_flat[:, None]
    derivative_weighted = derivative_flat * fit_weights_flat[:, None]

    right_hand_side = np.zeros((number_unknowns, number_rows), dtype=float)
    system_matrix = np.zeros((number_unknowns, number_unknowns), dtype=float)

    right_hand_side[:number_materials, :] = (log_projection_rows @ thickness_weighted).T
    system_matrix[:number_materials, :number_materials] = 2.0 * k0 * (thickness_flat.T @ thickness_weighted)

    if use_refraction:
        right_hand_side[number_materials:, :] = (log_projection_rows @ derivative_weighted).T
        system_matrix[:number_materials, number_materials:] = -j_ei * (thickness_flat.T @ derivative_weighted)
        system_matrix[number_materials:, :number_materials] = 2.0 * k0 * (derivative_flat.T @ thickness_weighted)
        system_matrix[number_materials:, number_materials:] = -j_ei * (derivative_flat.T @ derivative_weighted)

    scaled_matrix, scaled_rhs, _ = normalize_linear_system_determinant(system_matrix, right_hand_side)
    coefficient_estimates = -np.linalg.solve(scaled_matrix, scaled_rhs).T

    if not return_projections: return coefficient_estimates

    absorption_coefficients = coefficient_estimates[:, :number_materials]
    fitted_absorption_projection = -2.0 * k0 * np.einsum("rv,vtx->rtx", absorption_coefficients, thickness)

    if use_refraction:
        refraction_coefficients = coefficient_estimates[:, number_materials:]
        fitted_refraction_projection = j_ei * np.einsum("rv,vtx->rtx", refraction_coefficients, thickness_derivative)
    else:
        fitted_refraction_projection = np.zeros_like(fitted_absorption_projection)

    fitted_projection = fitted_absorption_projection + fitted_refraction_projection
    return coefficient_estimates, fitted_projection, fitted_absorption_projection, fitted_refraction_projection
'''
def estimate_material_coefficients(log_projection, thickness, thickness_derivative, j_ei, k0, return_projections=False):
    """
    Estimate material coefficients from log projections and material thickness sinograms.

    Parameters
    ----------
    log_projection : np.ndarray, shape (..., Ntheta, Nx)
        Log-domain projection data. Any leading dimensions are treated as independent rows.

    thickness : np.ndarray, shape (Nv, Ntheta, Nx)
        Material thickness sinograms. Nv is the number of material masks.

    thickness_derivative : np.ndarray, shape (Nv, Ntheta, Nx)
        Detector-axis derivative of the material thickness sinograms.

    j_ei : float
        Edge-illumination refraction sensitivity parameter. If zero, only absorption coefficients are estimated.

    k0 : float
        Wavenumber used in the absorption term.

    return_projections : bool, optional
        If True, also return the fitted total, absorption, and refraction projections.

    Returns
    -------
    coefficient_estimates : np.ndarray, shape (Nrow, Nv) or (Nrow, 2*Nv)
        Estimated material coefficients. The first Nv columns are the absorption coefficients. If j_ei is non-zero, the last Nv columns are the refraction coefficients.

    fitted_projection : np.ndarray, shape (Nrow, Ntheta, Nx), optional
        Fitted projection, returned only if return_projections=True.

    fitted_absorption_projection : np.ndarray, shape (Nrow, Ntheta, Nx), optional
        Absorption contribution, returned only if return_projections=True.

    fitted_refraction_projection : np.ndarray, shape (Nrow, Ntheta, Nx), optional
        Refraction contribution, returned only if return_projections=True.
    """

    log_projection = np.asarray(log_projection, dtype=float); thickness = np.asarray(thickness, dtype=float); thickness_derivative = np.asarray(thickness_derivative, dtype=float)
    if thickness.shape != thickness_derivative.shape: raise ValueError("thickness and thickness_derivative must have the same shape.")
    if thickness.ndim != 3: raise ValueError("thickness must have shape (Nv, Ntheta, Nx).")

    number_materials = thickness.shape[0]; sample_shape = thickness.shape[1:]
    if log_projection.shape[-2:] != sample_shape: raise ValueError("The last two dimensions of log_projection must match thickness.shape[1:].")

    log_projection_rows = log_projection.reshape(-1, np.prod(sample_shape))  # shape (Nrow, Ntheta*Nx)
    number_rows = log_projection_rows.shape[0]; use_refraction = not np.isclose(j_ei, 0.0)
    number_unknowns = 2 * number_materials if use_refraction else number_materials

    thickness_flat = thickness.reshape(number_materials, -1).T  # shape (Ntheta*Nx, Nv)
    derivative_flat = thickness_derivative.reshape(number_materials, -1).T  # shape (Ntheta*Nx, Nv)

    right_hand_side = np.zeros((number_unknowns, number_rows), dtype=float)
    system_matrix = np.zeros((number_unknowns, number_unknowns), dtype=float)

    right_hand_side[:number_materials, :] = (log_projection_rows @ thickness_flat).T
    system_matrix[:number_materials, :number_materials] = 2.0 * k0 * (thickness_flat.T @ thickness_flat)

    if use_refraction:
        right_hand_side[number_materials:, :] = (log_projection_rows @ derivative_flat).T
        system_matrix[:number_materials, number_materials:] = -j_ei * (thickness_flat.T @ derivative_flat)
        system_matrix[number_materials:, :number_materials] = 2.0 * k0 * (derivative_flat.T @ thickness_flat)
        system_matrix[number_materials:, number_materials:] = -j_ei * (derivative_flat.T @ derivative_flat)

       
    scaled_matrix, scaled_rhs, _ = normalize_linear_system_determinant(system_matrix, right_hand_side)
    coefficient_estimates = -np.linalg.solve(scaled_matrix, scaled_rhs).T
 
    if not return_projections: return coefficient_estimates

    absorption_coefficients = coefficient_estimates[:, :number_materials]
    fitted_absorption_projection = -2.0 * k0 * np.einsum("rv,vtx->rtx", absorption_coefficients, thickness)

    if use_refraction:
        refraction_coefficients = coefficient_estimates[:, number_materials:]
        fitted_refraction_projection = j_ei * np.einsum("rv,vtx->rtx", refraction_coefficients, thickness_derivative)
    else:
        fitted_refraction_projection = np.zeros_like(fitted_absorption_projection)

    fitted_projection = fitted_absorption_projection + fitted_refraction_projection

    return coefficient_estimates, fitted_projection, fitted_absorption_projection, fitted_refraction_projection
'''

def estimate_material_coefficients_3D(log_projection_3D, thickness_3D, thickness_derivative_3D, j_ei, k0, return_projections=False):
    """
    Estimate one common material-coefficient vector from multiple slices.

    log_projection_3D : np.ndarray, shape (Ns, Ntheta, Nx)
        Log-domain projection data for all slices.
    thickness_3D : np.ndarray, shape (Nv, Ntheta, Nx, Ns)
        Material thickness sinograms for all slices.
    thickness_derivative_3D : np.ndarray, shape (Nv, Ntheta, Nx, Ns)
        Detector-axis derivatives of thickness_3D.
    j_ei : float
        Edge-illumination refraction sensitivity parameter.
    k0 : float
        Wavenumber used in the absorption term.
    return_projections : bool, optional
        If True, also return fitted projections with shape (Ns, Ntheta, Nx).

    Returns
    -------
    coefficient_estimates : np.ndarray, shape (1, Nv) or (1, 2*Nv)
        One common beta/delta vector fitted using all slices together.
    """

    log_projection_3D = np.asarray(log_projection_3D, dtype=float); thickness_3D = np.asarray(thickness_3D, dtype=float); thickness_derivative_3D = np.asarray(thickness_derivative_3D, dtype=float)
    if thickness_3D.shape != thickness_derivative_3D.shape: raise ValueError("thickness_3D and thickness_derivative_3D must have the same shape.")
    if log_projection_3D.ndim != 3: raise ValueError("log_projection_3D must have shape (Ns, Ntheta, Nx).")
    if thickness_3D.ndim != 4: raise ValueError("thickness_3D must have shape (Nv, Ntheta, Nx, Ns).")

    n_slices, n_angles, n_detectors = log_projection_3D.shape
    number_materials = thickness_3D.shape[0]
    if thickness_3D.shape[1:] != (n_angles, n_detectors, n_slices): raise ValueError("thickness_3D must have shape (Nv, Ntheta, Nx, Ns), matching log_projection_3D.")

    log_projection_fit = np.transpose(log_projection_3D, (1, 2, 0))  # shape (Ntheta, Nx, Ns)
    log_projection_row = log_projection_fit.reshape(1, -1)  # shape (1, Ntheta*Nx*Ns)

    use_refraction = not np.isclose(j_ei, 0.0)
    number_unknowns = 2 * number_materials if use_refraction else number_materials

    thickness_flat = thickness_3D.reshape(number_materials, -1).T
    derivative_flat = thickness_derivative_3D.reshape(number_materials, -1).T

    right_hand_side = np.zeros((number_unknowns, 1), dtype=float)
    system_matrix = np.zeros((number_unknowns, number_unknowns), dtype=float)

    right_hand_side[:number_materials, :] = (log_projection_row @ thickness_flat).T
    system_matrix[:number_materials, :number_materials] = 2.0 * k0 * (thickness_flat.T @ thickness_flat)

    if use_refraction:
        right_hand_side[number_materials:, :] = (log_projection_row @ derivative_flat).T
        system_matrix[:number_materials, number_materials:] = -j_ei * (thickness_flat.T @ derivative_flat)
        system_matrix[number_materials:, :number_materials] = 2.0 * k0 * (derivative_flat.T @ thickness_flat)
        system_matrix[number_materials:, number_materials:] = -j_ei * (derivative_flat.T @ derivative_flat)

    scaled_matrix, scaled_rhs, _ = normalize_linear_system_determinant(system_matrix, right_hand_side)
    coefficient_estimates = -np.linalg.solve(scaled_matrix, scaled_rhs).T

    if not return_projections: return coefficient_estimates

    absorption_coefficients = coefficient_estimates[:, :number_materials]
    fitted_absorption_projection = -2.0 * k0 * np.einsum("rv,vtxs->txs", absorption_coefficients, thickness_3D)[0]

    if use_refraction:
        refraction_coefficients = coefficient_estimates[:, number_materials:]
        fitted_refraction_projection = j_ei * np.einsum("rv,vtxs->txs", refraction_coefficients, thickness_derivative_3D)[0]
    else:
        fitted_refraction_projection = np.zeros_like(fitted_absorption_projection)

    fitted_projection = fitted_absorption_projection + fitted_refraction_projection

    return coefficient_estimates, np.transpose(fitted_projection, (2, 0, 1)), np.transpose(fitted_absorption_projection, (2, 0, 1)), np.transpose(fitted_refraction_projection, (2, 0, 1))

def compose_ljr_projection_components(coefficient_estimates, thickness_sinograms):
    """
    Build absorption and refraction-integral projection components from LJR coefficients.

    Parameters
    ----------
    coefficient_estimates : np.ndarray, shape (Nrow, 2*Nv) or (2*Nv,)
        Estimated material coefficients. First Nv entries are beta-like coefficients, last Nv entries are delta-like coefficients.
    thickness_sinograms : np.ndarray, shape (Nv, Ntheta, Nx)
        Material thickness sinograms.

    Returns
    -------
    projection_refraction_integral : np.ndarray, shape (Nrow, Ntheta, Nx) or (Ntheta, Nx)
        Weighted sum of thickness_sinograms using the delta-like coefficients.
    projection_absorption : np.ndarray, shape (Nrow, Ntheta, Nx) or (Ntheta, Nx)
        Weighted sum of thickness_sinograms using the beta-like coefficients.
    """

    coefficient_estimates = np.asarray(coefficient_estimates, dtype=float); thickness_sinograms = np.asarray(thickness_sinograms, dtype=float)
    if coefficient_estimates.ndim == 1: coefficient_estimates = coefficient_estimates[None, :]
    number_materials = thickness_sinograms.shape[0]
    if coefficient_estimates.shape[-1] != 2 * number_materials: raise ValueError("coefficient_estimates must have shape (Nrow, 2*Nv), with Nv = thickness_sinograms.shape[0].")

    beta_coefficients = coefficient_estimates[:, :number_materials]
    delta_coefficients = coefficient_estimates[:, number_materials:]

    projection_absorption = np.einsum("rv,vtx->rtx", beta_coefficients, thickness_sinograms)
    projection_refraction_integral = np.einsum("rv,vtx->rtx", delta_coefficients, thickness_sinograms)

    return np.squeeze(projection_refraction_integral), np.squeeze(projection_absorption)


def compose_ljr_projection_components_3D(coefficient_estimates, thickness_sinograms_3D):
    coefficient_estimates = np.asarray(coefficient_estimates, dtype=float); thickness_sinograms_3D = np.asarray(thickness_sinograms_3D, dtype=float)
    if coefficient_estimates.ndim == 1: coefficient_estimates = coefficient_estimates[None, :]
    if thickness_sinograms_3D.ndim != 4: raise ValueError("thickness_sinograms_3D must have shape (Nv, Ntheta, Nx, Ns).")

    number_materials = thickness_sinograms_3D.shape[0]
    if coefficient_estimates.shape[-1] != 2 * number_materials: raise ValueError("coefficient_estimates must have shape (Nrow, 2*Nv).")

    beta_coefficients = coefficient_estimates[:, :number_materials]
    delta_coefficients = coefficient_estimates[:, number_materials:]

    projection_absorption_3D = np.einsum("rv,vtxs->rtxs", beta_coefficients, thickness_sinograms_3D)
    projection_refraction_integral_3D = np.einsum("rv,vtxs->rtxs", delta_coefficients, thickness_sinograms_3D)

    return np.squeeze(projection_refraction_integral_3D), np.squeeze(projection_absorption_3D)