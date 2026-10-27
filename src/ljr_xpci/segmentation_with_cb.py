import numpy as np
import astra
from scipy.ndimage import shift
from .preprocessing import filter_ft
from .diemoz_filter import apply_single_shot_ei_filter
import numpy as np
from .radon_tools import astra_iradon_parallel, astra_iradon_conebeam_3D
from ljr_xpci.utility_functions import err_n
from .utility_functions import err_n
from .utility_functions import moving_mean_2d, moving_mean_3d, normalise_filter_fraction

def ljr_segment(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_window_size=7):
    """
    Segment a parallel-beam log-projection dataset into material masks for LJR.

    Parameters
    ----------
    log_projection_noise : np.ndarray, shape (Ns, Ntheta, Nx)
        Noisy log-domain projection dataset. Axis 0 is slice, axis 1 is projection angle, axis 2 is detector x.
    detector_pixel_size : float
        Detector pixel size used to scale reconstructed values.
    gamma_jei : float
        Single-shot edge-illumination retrieval parameter.
    noise_mask : np.ndarray, shape (Nx, Nx)
        Mask used to estimate the noise-driven minimum threshold.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    mask_container : np.ndarray, shape (Nx, Nx), optional
        Optional external/container mask.
    projection_shift : float
        Detector-axis shift applied to each single-shot sinogram before reconstruction.
    threshold_step : float
        Threshold step.
    filter_fraction_by_axis : float or sequence
        Fourier filtering fraction. If scalar, the same value is used for all axes. If sequence, it follows Python order (Ns, Ntheta, Nx).
    window_density_threshold : float, optional
        Minimum fraction of changed pixels required inside a local moving_window_size × moving_window_size window to accept a new segmentation threshold. Default is 0.75.    
    n_std : float
        Noise threshold multiplier.
    moving_window_size : int
        Moving-mean window size.

    Returns
    -------
    material_masks : list[np.ndarray]
        Material masks, each with shape (Nx, Nx).
    """

    test_mat = False;
    if test_mat:
        from scipy.io import loadmat
        test_data = loadmat("/home/amazzolani/Matlab_drive_scripts/remotedata_Andrea_clio/pd/Khush_folder/LJR_segment_internal_test.mat")

    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ns, Ntheta, Nx).")

    n_slices, n_angles, n_detectors = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")
    if noise_mask.shape != (n_detectors, n_detectors): raise ValueError("noise_mask must have shape (Nx, Nx).")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    max_mean_window_count = window_density_threshold * moving_window_size**2
    if mask_container is None: mask_container = np.zeros_like(noise_mask, dtype=bool)
    else: mask_container = np.asarray(mask_container).astype(bool)

    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))
    central_slice_index = int(np.ceil(n_slices / 2.0) - 1)

    if test_mat:
        noisy_reconstruction = test_data["s_noise"].astype(float)
        filtered_reconstruction = test_data["s_Ln_1"].astype(float)
    else:
        noisy_reconstruction = astra_iradon_parallel(log_projection_noise[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size
        
        filtered_reconstruction = astra_iradon_parallel(filtered_log_projection[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size

    masked_noise_reconstruction = noisy_reconstruction * noise_mask

    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

    candidate_thresholds = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds = candidate_thresholds[candidate_thresholds >= minimum_threshold]

    if candidate_thresholds.size == 0: return []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)
          

    single_shot_reconstructions = np.zeros((n_slices, n_detectors, n_detectors), dtype=float)
    for slice_index in range(n_slices):
        shifted_sinogram = shift(single_shot_projection[slice_index], shift=(0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
        single_shot_reconstructions[slice_index] = astra_iradon_parallel(shifted_sinogram, theta_degrees, n_detectors) / detector_pixel_size
    
    # print("Python ST size =", single_shot_reconstructions.shape)
    st_mean = np.mean(single_shot_reconstructions, axis=(1, 2)); st_min = np.min(single_shot_reconstructions, axis=(1, 2)); st_max = np.max(single_shot_reconstructions, axis=(1, 2))
    # print("Python ST mean/min/max =", st_mean, st_min, st_max)      
          
    s1_tmp = np.squeeze(np.mean(single_shot_reconstructions, axis=0))
    if test_mat:
        s1_tmp = test_data["s1_tmp"].astype(float) #?

    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool)
    threshold_index = 0

    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            current_threshold = candidate_thresholds[threshold_index]; threshold_index += 1
            current_condition = s1_tmp < current_threshold
            condition_difference = current_condition.astype(float) - previous_condition.astype(float)
            local_change_count = np.max(moving_mean_2d(condition_difference, moving_window_size)) * moving_window_size**2

            if local_change_count > max_mean_window_count:
                selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy(); continue_search = False
            if np.sum(current_condition) >= n_detectors**2 - moving_window_size**2: continue_search = False
                
    if len(selected_conditions) == 0: return []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds[-1])

    material_masks = [filtered_reconstruction < selected_thresholds[0]]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool))

    for mask_index in range(len(material_masks)):
        if not np.any(material_masks[mask_index]): material_masks[mask_index] = np.zeros_like(material_masks[0], dtype=bool)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks,selected_thresholds

 
                         
def ljr_segment_3D(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_kernel_size_3d=(7, 7, None)):
    """
    Segment a stack of adjacent slices into material masks for LJR.

    The threshold bins are selected using all input slices together, rather than the mean slice only. A new threshold is accepted when at least one local 3D window, spanning all selected slices and moving_window_size × moving_window_size pixels in each slice, contains a fraction of newly selected pixels larger than window_density_threshold.

    Parameters
    ----------
    log_projection_noise : np.ndarray, shape (Ns, Ntheta, Nx)
        Noisy log-domain projection dataset. Axis 0 is slice, axis 1 is projection angle, axis 2 is detector x.
    detector_pixel_size : float
        Detector pixel size used to scale reconstructed values.
    gamma_jei : float
        Single-shot edge-illumination retrieval parameter.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    noise_mask : np.ndarray, shape (Nx, Nx)
        Background mask used to estimate the noise threshold as n_std × std outside the sample.
    mask_container : np.ndarray, shape (Nx, Nx) or (Ns, Nx, Nx), optional
        Optional container mask.
    projection_shift : float
        Detector-axis shift applied to each single-shot sinogram before reconstruction.
    threshold_step : float
        Intensity increment used to build the candidate threshold grid for material-mask selection.
    filter_fraction_by_axis : float or sequence
        Fourier filtering fraction. If sequence, it follows Python order (Ns, Ntheta, Nx).
    window_density_threshold : float
        Minimum fraction of newly selected pixels required in at least one Ns × moving_window_size × moving_window_size local window to accept the current threshold as a new material bin.
    n_std : float
        Multiplier used to define the noise threshold.
    moving_window_size : int
        Side length, in pixels, of the local square window used in each slice.

    Returns
    -------
    material_masks : list[np.ndarray]
        Material masks, each with shape (Ns, Nx, Nx).
    selected_thresholds : list[float]
        Selected intensity thresholds.
    """
    
    window_x, window_y, window_z = moving_kernel_size_3d
    window_x = int(window_x); window_y = int(window_y); window_z = 1 if window_z is None else int(window_z)
    moving_kernel_size_3d = (window_x, window_y, window_z)
    window_volume = window_x * window_y * window_z

    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ns, Ntheta, Nx).")

    n_slices, n_angles, n_detectors = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")
    if noise_mask.shape != (n_detectors, n_detectors): raise ValueError("noise_mask must have shape (Nx, Nx).")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    if mask_container is None: mask_container = np.zeros((n_slices, n_detectors, n_detectors), dtype=bool)
    else:
        mask_container = np.asarray(mask_container).astype(bool)
        if mask_container.ndim == 2: mask_container = np.broadcast_to(mask_container, (n_slices, n_detectors, n_detectors)).copy()
            
    segmentation_domain = ~mask_container if np.any(mask_container) else np.ones_like(mask_container, dtype=bool)
    n_domain_pixels = np.count_nonzero(segmentation_domain)

    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))

    noisy_reconstructions = np.zeros((n_slices, n_detectors, n_detectors), dtype=float)
    filtered_reconstructions = np.zeros_like(noisy_reconstructions)
    for slice_index in range(n_slices):
        noisy_reconstructions[slice_index] = astra_iradon_parallel(log_projection_noise[slice_index], theta_degrees, n_detectors) / detector_pixel_size
        filtered_reconstructions[slice_index] = astra_iradon_parallel(filtered_log_projection[slice_index], theta_degrees, n_detectors) / detector_pixel_size

    masked_noise_reconstruction = noisy_reconstructions * noise_mask[None, :, :]
    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

    candidate_thresholds = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds = candidate_thresholds[candidate_thresholds >= minimum_threshold]
    if candidate_thresholds.size == 0: return [], []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)

    single_shot_reconstructions = np.zeros((n_slices, n_detectors, n_detectors), dtype=float)
    for slice_index in range(n_slices):
        shifted_sinogram = shift(single_shot_projection[slice_index], shift=(0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
        single_shot_reconstructions[slice_index] = astra_iradon_parallel(shifted_sinogram, theta_degrees, n_detectors) / detector_pixel_size

    s1_tmp = single_shot_reconstructions
    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool)
    threshold_index = 0

    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            current_threshold = candidate_thresholds[threshold_index]; threshold_index += 1
            current_condition = (s1_tmp < current_threshold) & segmentation_domain
            condition_difference = current_condition.astype(float) - previous_condition.astype(float)
            local_change_density = np.max(moving_mean_3d(condition_difference, moving_kernel_size_3d))

            if local_change_density > window_density_threshold:
                selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy(); continue_search = False
            if np.sum(current_condition) >= current_condition.size - window_volume: continue_search = False

    if len(selected_conditions) == 0: return [], []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds[-1])

    material_masks = [filtered_reconstructions < selected_thresholds[0]]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool))

    for mask_index in range(len(material_masks)):
        if not np.any(material_masks[mask_index]): material_masks[mask_index] = np.zeros_like(material_masks[0], dtype=bool)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks, selected_thresholds


def ljr_segment_conebeam(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, cone_geometry, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_window_size=7, gpu_index=0):
    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ndet_y, Ntheta, Ndet_x).")

    n_detector_y, n_angles, n_detector_x = log_projection_noise.shape
    volume_shape = tuple(int(v) for v in cone_geometry["volume_shape"])  # (Nz, Ny, Nx)
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    if noise_mask.ndim == 2: noise_mask = np.broadcast_to(noise_mask[None, :, :], volume_shape).copy()
    if noise_mask.shape != volume_shape: raise ValueError("noise_mask must have shape (Ny, Nx) or (Nz, Ny, Nx).")

    if mask_container is None: mask_container = np.zeros(volume_shape, dtype=bool)
    else:
        mask_container = np.asarray(mask_container).astype(bool)
        if mask_container.ndim == 2: mask_container = np.broadcast_to(mask_container[None, :, :], volume_shape).copy()
        if mask_container.shape != volume_shape: raise ValueError("mask_container must have shape (Ny, Nx) or (Nz, Ny, Nx).")

    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))

    noisy_reconstruction = astra_iradon_conebeam_3D(log_projection_noise, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    filtered_reconstruction = astra_iradon_conebeam_3D(filtered_log_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size

    masked_noise_reconstruction = noisy_reconstruction * noise_mask
    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

    candidate_thresholds = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds = candidate_thresholds[candidate_thresholds >= minimum_threshold]
    if candidate_thresholds.size == 0: return [], []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)
    shifted_projection = shift(single_shot_projection, shift=(0.0, 0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
    s1_tmp = astra_iradon_conebeam_3D(shifted_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    
    print("conebeam s1_tmp mean/min/max/std =", np.mean(s1_tmp), np.min(s1_tmp), np.max(s1_tmp), np.std(s1_tmp, ddof=1))
    print("conebeam minimum_threshold =", minimum_threshold)
    print("conebeam candidate thresholds first/last/n =", candidate_thresholds[0], candidate_thresholds[-1], candidate_thresholds.size)

    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool)
    threshold_index = 0

    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            current_threshold = candidate_thresholds[threshold_index]; threshold_index += 1
            current_condition = s1_tmp < current_threshold
            condition_difference = current_condition.astype(float) - previous_condition.astype(float)
            local_change_density = np.max(moving_mean_3d(condition_difference, moving_window_size))

            if local_change_density > window_density_threshold:
                selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy(); continue_search = False
            if np.sum(current_condition) >= current_condition.size - volume_shape[0] * moving_window_size**2: continue_search = False

    if len(selected_conditions) == 0: return [], []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds[-1])

    material_masks = [filtered_reconstruction < selected_thresholds[0]]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool))

    for mask_index in range(len(material_masks)):
        if not np.any(material_masks[mask_index]): material_masks[mask_index] = np.zeros_like(material_masks[0], dtype=bool)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks, selected_thresholds
