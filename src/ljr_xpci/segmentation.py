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
from concurrent.futures import ThreadPoolExecutor

    
def ljr_segment(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_window_size=7):
    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ns, Ntheta, Nx).")

    n_slices, n_angles, n_detectors = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")
    if noise_mask.shape != (n_detectors, n_detectors): raise ValueError("noise_mask must have shape (Nx, Nx).")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    moving_window_size = int(moving_window_size); max_mean_window_count = window_density_threshold * moving_window_size**2
    mask_container = np.zeros_like(noise_mask, dtype=bool) if mask_container is None else np.asarray(mask_container, dtype=bool)
    if mask_container.shape != (n_detectors, n_detectors): raise ValueError("mask_container must have shape (Nx, Nx).")

    segmentation_domain = ~mask_container if np.any(mask_container) else np.ones((n_detectors, n_detectors), dtype=bool)
    n_domain_pixels = np.count_nonzero(segmentation_domain)

    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))
    central_slice_index = int(np.ceil(n_slices / 2.0) - 1)

    noisy_reconstruction = astra_iradon_parallel(log_projection_noise[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size
    filtered_reconstruction = astra_iradon_parallel(filtered_log_projection[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size

    masked_noise_reconstruction = noisy_reconstruction * noise_mask
    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

    candidate_thresholds_full = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds_full = candidate_thresholds_full[candidate_thresholds_full >= minimum_threshold]
    if candidate_thresholds_full.size == 0: return [], []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)

    single_shot_reconstructions = np.zeros((n_slices, n_detectors, n_detectors), dtype=float)
    for slice_index in range(n_slices):
        shifted_sinogram = shift(single_shot_projection[slice_index], shift=(0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
        single_shot_reconstructions[slice_index] = astra_iradon_parallel(shifted_sinogram, theta_degrees, n_detectors) / detector_pixel_size

    s1_tmp = np.squeeze(np.mean(single_shot_reconstructions, axis=0))

    domain_values = s1_tmp[segmentation_domain]
    change_indices = np.searchsorted(candidate_thresholds_full, domain_values.ravel(), side="right")
    change_indices = np.unique(change_indices[(change_indices >= 0) & (change_indices < candidate_thresholds_full.size)])
    candidate_thresholds = candidate_thresholds_full[change_indices]
    if candidate_thresholds.size == 0: candidate_thresholds = candidate_thresholds_full[-1:]

    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool)
    threshold_index = 0

    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            current_threshold = candidate_thresholds[threshold_index]; threshold_index += 1
            current_condition = (s1_tmp < current_threshold) & segmentation_domain
            condition_difference = current_condition.astype(float) - previous_condition.astype(float)
            local_change_count = np.max(moving_mean_2d(condition_difference, moving_window_size)) * moving_window_size**2

            if local_change_count > max_mean_window_count:
                selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy(); continue_search = False
            if np.sum(current_condition) >= n_domain_pixels - moving_window_size**2: continue_search = False

    if len(selected_conditions) == 0: return [], []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds_full[-1])

    material_masks = [(filtered_reconstruction < selected_thresholds[0]) & segmentation_domain]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool) & segmentation_domain)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks, selected_thresholds

def ljr_segment_3D(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_kernel_size_3d=(7, 7, None)):
    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ns, Ntheta, Nx).")

    n_slices, n_angles, n_detectors = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")
    if noise_mask.shape != (n_detectors, n_detectors): raise ValueError("noise_mask must have shape (Nx, Nx).")

    window_x, window_y, window_z = moving_kernel_size_3d
    window_x = int(window_x); window_y = int(window_y); window_z = n_slices if window_z is None else min(int(window_z), n_slices)
    if min(window_x, window_y, window_z) < 1: raise ValueError("All entries of moving_kernel_size_3d must be positive.")
    moving_kernel_size_3d = (window_x, window_y, window_z); window_volume = window_x * window_y * window_z

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    mask_container = np.zeros((n_slices, n_detectors, n_detectors), dtype=bool) if mask_container is None else np.asarray(mask_container, dtype=bool)
    if mask_container.ndim == 2: mask_container = np.broadcast_to(mask_container, (n_slices, n_detectors, n_detectors)).copy()
    if mask_container.shape != (n_slices, n_detectors, n_detectors): raise ValueError("mask_container must have shape (Ns, Nx, Nx) or (Nx, Nx).")

    segmentation_domain = ~mask_container if np.any(mask_container) else np.ones((n_slices, n_detectors, n_detectors), dtype=bool)
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

    candidate_thresholds_full = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds_full = candidate_thresholds_full[candidate_thresholds_full >= minimum_threshold]
    if candidate_thresholds_full.size == 0: return [], []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)

    single_shot_reconstructions = np.zeros((n_slices, n_detectors, n_detectors), dtype=float)
    for slice_index in range(n_slices):
        shifted_sinogram = shift(single_shot_projection[slice_index], shift=(0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
        single_shot_reconstructions[slice_index] = astra_iradon_parallel(shifted_sinogram, theta_degrees, n_detectors) / detector_pixel_size

    s1_tmp = single_shot_reconstructions

    domain_values = s1_tmp[segmentation_domain]
    change_indices = np.searchsorted(candidate_thresholds_full, domain_values.ravel(), side="right")
    change_indices = np.unique(change_indices[(change_indices >= 0) & (change_indices < candidate_thresholds_full.size)])
    candidate_thresholds = candidate_thresholds_full[change_indices]
    if candidate_thresholds.size == 0: candidate_thresholds = candidate_thresholds_full[-1:]

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
            if np.sum(current_condition) >= n_domain_pixels - window_volume: continue_search = False

    if len(selected_conditions) == 0: return [], []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds_full[-1])

    material_masks = [(filtered_reconstructions < selected_thresholds[0]) & segmentation_domain]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool) & segmentation_domain)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks, selected_thresholds

def ljr_segment_conebeam(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, cone_geometry, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_kernel_size_3d=(7, 7, None), gpu_index=0):
    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ndet_y, Ntheta, Ndet_x).")
    if threshold_step <= 0: raise ValueError("threshold_step must be positive.")

    n_detector_y, n_angles, n_detector_x = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))

    noisy_reconstructions = astra_iradon_conebeam_3D(log_projection_noise, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    filtered_reconstructions = astra_iradon_conebeam_3D(filtered_log_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    volume_shape = noisy_reconstructions.shape

    window_x, window_y, window_z = moving_kernel_size_3d
    window_x = int(window_x); window_y = int(window_y); window_z = volume_shape[0] if window_z is None else min(int(window_z), volume_shape[0])
    if min(window_x, window_y, window_z) < 1: raise ValueError("All entries of moving_kernel_size_3d must be positive.")
    moving_kernel_size_3d = (window_x, window_y, window_z); window_volume = window_x * window_y * window_z

    if noise_mask.ndim == 2: noise_mask = np.broadcast_to(noise_mask[None, :, :], volume_shape).copy()
    if noise_mask.shape != volume_shape: raise ValueError("noise_mask must have shape (Ny, Nx) or match the cone-beam reconstruction shape.")

    mask_container = np.zeros(volume_shape, dtype=bool) if mask_container is None else np.asarray(mask_container, dtype=bool)
    if mask_container.ndim == 2: mask_container = np.broadcast_to(mask_container[None, :, :], volume_shape).copy()
    if mask_container.shape != volume_shape: raise ValueError("mask_container must have shape (Ny, Nx) or match the cone-beam reconstruction shape.")

    segmentation_domain = ~mask_container if np.any(mask_container) else np.ones(volume_shape, dtype=bool)
    n_domain_pixels = np.count_nonzero(segmentation_domain)

    masked_noise_reconstruction = noisy_reconstructions * noise_mask
    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

    candidate_thresholds_full = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds_full = candidate_thresholds_full[candidate_thresholds_full >= minimum_threshold]
    if candidate_thresholds_full.size == 0: return [], []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)
    shifted_projection = shift(single_shot_projection, shift=(0.0, 0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
    s1_tmp = astra_iradon_conebeam_3D(shifted_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size

    domain_values = s1_tmp[segmentation_domain]
    change_indices = np.searchsorted(candidate_thresholds_full, domain_values.ravel(), side="right")
    change_indices = np.unique(change_indices[(change_indices >= 0) & (change_indices < candidate_thresholds_full.size)])
    candidate_thresholds = candidate_thresholds_full[change_indices]
    if candidate_thresholds.size == 0: candidate_thresholds = candidate_thresholds_full[-1:]

    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool); threshold_index = 0

    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            current_threshold = candidate_thresholds[threshold_index]; threshold_index += 1
            current_condition = (s1_tmp < current_threshold) & segmentation_domain
            condition_difference = current_condition.astype(float) - previous_condition.astype(float)
            local_change_density = np.max(moving_mean_3d(condition_difference, moving_kernel_size_3d))

            if local_change_density > window_density_threshold:
                selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy(); continue_search = False
            if np.sum(current_condition) >= n_domain_pixels - window_volume: continue_search = False

    if len(selected_conditions) == 0: return [], []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds_full[-1])

    material_masks = [(filtered_reconstructions < selected_thresholds[0]) & segmentation_domain]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool) & segmentation_domain)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks, selected_thresholds

def ljr_segment_conebeam_bisection(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, cone_geometry, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_kernel_size_2d=7, moving_mean_backend="auto", parallel_workers=None, gpu_index=0):
    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ndet_y, Ntheta, Ndet_x).")
    if threshold_step <= 0: raise ValueError("threshold_step must be positive.")

    moving_kernel_size_2d = int(moving_kernel_size_2d)
    if moving_kernel_size_2d < 1: raise ValueError("moving_kernel_size_2d must be positive.")
    window_area = moving_kernel_size_2d**2

    n_detector_y, n_angles, n_detector_x = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))

    noisy_reconstructions = astra_iradon_conebeam_3D(log_projection_noise, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    filtered_reconstructions = astra_iradon_conebeam_3D(filtered_log_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    volume_shape = noisy_reconstructions.shape

    if noise_mask.ndim == 2: noise_mask = np.broadcast_to(noise_mask[None, :, :], volume_shape).copy()
    if noise_mask.shape != volume_shape: raise ValueError("noise_mask must have shape (Ny, Nx) or match the cone-beam reconstruction shape.")

    mask_container = np.zeros(volume_shape, dtype=bool) if mask_container is None else np.asarray(mask_container, dtype=bool)
    if mask_container.ndim == 2: mask_container = np.broadcast_to(mask_container[None, :, :], volume_shape).copy()
    if mask_container.shape != volume_shape: raise ValueError("mask_container must have shape (Ny, Nx) or match the cone-beam reconstruction shape.")

    segmentation_domain = ~mask_container if np.any(mask_container) else np.ones(volume_shape, dtype=bool)
    n_domain_pixels = np.count_nonzero(segmentation_domain)

    masked_noise_reconstruction = noisy_reconstructions * noise_mask
    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

    candidate_thresholds_full = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds_full = candidate_thresholds_full[candidate_thresholds_full >= minimum_threshold]
    if candidate_thresholds_full.size == 0: return [], []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)
    shifted_projection = shift(single_shot_projection, shift=(0.0, 0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
    s1_tmp = astra_iradon_conebeam_3D(shifted_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size

    domain_values = s1_tmp[segmentation_domain]
    change_indices = np.searchsorted(candidate_thresholds_full, domain_values.ravel(), side="right")
    change_indices = np.unique(change_indices[(change_indices >= 0) & (change_indices < candidate_thresholds_full.size)])
    candidate_thresholds = candidate_thresholds_full[change_indices]
    if candidate_thresholds.size == 0: candidate_thresholds = candidate_thresholds_full[-1:]

    def max_moving_density_2d_over_slices(condition_difference):
        # print("trying moving_mean backend =", moving_mean_backend)
        if moving_mean_backend in ("auto", "gpu"):
            try:
                import cupy as cp
                #print("using GPU CuPy moving mean")
                from cupyx.scipy.ndimage import uniform_filter as cupy_uniform_filter
                with cp.cuda.Device(gpu_index):
                    condition_gpu = cp.asarray(condition_difference, dtype=cp.float32)
                    local_sum = cupy_uniform_filter(condition_gpu, size=(1, moving_kernel_size_2d, moving_kernel_size_2d), mode="constant", cval=0.0) * window_area
                    local_count = cupy_uniform_filter(cp.ones_like(condition_gpu), size=(1, moving_kernel_size_2d, moving_kernel_size_2d), mode="constant", cval=0.0) * window_area
                    return float(cp.max(local_sum / local_count).get())
            except Exception:
                if moving_mean_backend == "gpu": raise

        def slice_density(slice_data):
            return float(np.max(moving_mean_2d(slice_data, moving_kernel_size_2d)))

        if parallel_workers is None:
            max_workers = min(condition_difference.shape[0], 8)
        else:
            max_workers = int(parallel_workers)

        if max_workers <= 1:
            local_change_density = 0.0
            for slice_index in range(condition_difference.shape[0]):
                local_change_density = max(local_change_density, slice_density(condition_difference[slice_index]))
                if local_change_density > window_density_threshold: break
            return local_change_density
        #print("CPU parallel active, workers =", max_workers, "slices =", condition_difference.shape[0])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return max(executor.map(slice_density, condition_difference))
        
    def closest_candidate_index(left_index, right_index, target_threshold):
        candidate_index = np.searchsorted(candidate_thresholds, target_threshold, side="left")
        candidate_index = min(max(candidate_index, left_index), right_index)
        if candidate_index > left_index and abs(candidate_thresholds[candidate_index - 1] - target_threshold) <= abs(candidate_thresholds[candidate_index] - target_threshold): candidate_index -= 1
        return candidate_index

    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool); threshold_index = 0
    last_threshold_jump = threshold_step

    while threshold_index < candidate_thresholds.size:
        search_start_index = threshold_index
        condition_cache = {}

        def evaluate_threshold(candidate_index):
            if candidate_index not in condition_cache:
                current_threshold = candidate_thresholds[candidate_index]
                current_condition = (s1_tmp < current_threshold) & segmentation_domain
                condition_difference = current_condition.astype(float) - previous_condition.astype(float)
                local_change_density = max_moving_density_2d_over_slices(condition_difference)
                condition_cache[candidate_index] = local_change_density > window_density_threshold, current_condition, current_threshold
            return condition_cache[candidate_index]

        trial_jump = max(last_threshold_jump, threshold_step)
        trial_index = closest_candidate_index(search_start_index, candidate_thresholds.size - 1, candidate_thresholds[search_start_index] + trial_jump)
        accepted, _, _ = evaluate_threshold(trial_index)

        lower_index = search_start_index
        while not accepted and trial_index < candidate_thresholds.size - 1:
            lower_index = trial_index + 1
            trial_jump *= 2.0
            next_trial_index = closest_candidate_index(lower_index, candidate_thresholds.size - 1, candidate_thresholds[search_start_index] + trial_jump)
            if next_trial_index <= trial_index: next_trial_index = min(trial_index + 1, candidate_thresholds.size - 1)
            trial_index = next_trial_index
            accepted, _, _ = evaluate_threshold(trial_index)

        if not accepted: break

        upper_index = trial_index
        while lower_index < upper_index:
            middle_threshold = 0.5 * (candidate_thresholds[lower_index] + candidate_thresholds[upper_index])
            middle_index = closest_candidate_index(lower_index, upper_index - 1, middle_threshold)
            accepted_middle, _, _ = evaluate_threshold(middle_index)

            if accepted_middle: upper_index = middle_index
            else: lower_index = middle_index + 1

        accepted_index = lower_index
        _, current_condition, current_threshold = evaluate_threshold(accepted_index)

        selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy()
        threshold_index = accepted_index + 1
        last_threshold_jump = max(current_threshold - candidate_thresholds[search_start_index], threshold_step)

    if len(selected_conditions) == 0: return [], []

    selected_conditions.append(selected_conditions[-1].copy())
    selected_thresholds.append(candidate_thresholds_full[-1])

    material_masks = [(filtered_reconstructions < selected_thresholds[0]) & segmentation_domain]
    for condition_index in range(1, len(selected_conditions)):
        material_masks.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool) & segmentation_domain)

    if np.any(mask_container):
        material_masks[-1] = mask_container
        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = 2 * material_masks[mask_index].astype(int) - material_masks[-1].astype(int) == 2
    else:
        material_masks = material_masks[:-1]

    return material_masks, selected_thresholds

def LJR_conebeam_ind(log_projection_noise, detector_pixel_size, gamma_jei, theta_degrees, noise_mask, cone_geometry, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, window_density_threshold=0.75, n_std=2.0, moving_kernel_size_2d=7, gpu_index=0):
    log_projection_noise = np.asarray(log_projection_noise); noise_mask = np.asarray(noise_mask)
    if log_projection_noise.ndim != 3: raise ValueError("log_projection_noise must have shape (Ndet_y, Ntheta, Ndet_x).")
    if threshold_step <= 0: raise ValueError("threshold_step must be positive.")

    moving_kernel_size_2d = int(moving_kernel_size_2d)
    if moving_kernel_size_2d < 1: raise ValueError("moving_kernel_size_2d must be positive.")

    n_detector_y, n_angles, n_detector_x = log_projection_noise.shape
    if len(theta_degrees) != n_angles: raise ValueError("len(theta_degrees) must match log_projection_noise.shape[1].")

    filter_fraction_by_axis = normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))

    noisy_reconstructions = astra_iradon_conebeam_3D(log_projection_noise, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    filtered_reconstructions = astra_iradon_conebeam_3D(filtered_log_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size
    volume_shape = noisy_reconstructions.shape

    if noise_mask.ndim == 2: noise_mask = np.broadcast_to(noise_mask[None, :, :], volume_shape).copy()
    if noise_mask.shape != volume_shape: raise ValueError("noise_mask must have shape (Ny, Nx) or match the cone-beam reconstruction shape.")

    mask_container = np.zeros(volume_shape, dtype=bool) if mask_container is None else np.asarray(mask_container, dtype=bool)
    if mask_container.ndim == 2: mask_container = np.broadcast_to(mask_container[None, :, :], volume_shape).copy()
    if mask_container.shape != volume_shape: raise ValueError("mask_container must have shape (Ny, Nx) or match the cone-beam reconstruction shape.")

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)
    shifted_projection = shift(single_shot_projection, shift=(0.0, 0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
    s1_tmp = astra_iradon_conebeam_3D(shifted_projection, theta_degrees, cone_geometry, gpu_index=gpu_index) / detector_pixel_size

    slice_masks = []
    selected_thresholds_by_slice = []

    for slice_index in range(volume_shape[0]):
        noisy_slice = noisy_reconstructions[slice_index]
        filtered_slice = filtered_reconstructions[slice_index]
        s1_slice = s1_tmp[slice_index]
        noise_mask_slice = noise_mask[slice_index]
        mask_container_slice = mask_container[slice_index]

        segmentation_domain = ~mask_container_slice if np.any(mask_container_slice) else np.ones_like(s1_slice, dtype=bool)
        n_domain_pixels = np.count_nonzero(segmentation_domain)

        masked_noise_reconstruction = noisy_slice * noise_mask_slice
        positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
        minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0

        candidate_thresholds_full = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
        candidate_thresholds_full = candidate_thresholds_full[candidate_thresholds_full >= minimum_threshold]

        if candidate_thresholds_full.size == 0 or n_domain_pixels == 0:
            slice_masks.append([])
            selected_thresholds_by_slice.append([])
            continue

        domain_values = s1_slice[segmentation_domain]
        change_indices = np.searchsorted(candidate_thresholds_full, domain_values.ravel(), side="right")
        change_indices = np.unique(change_indices[(change_indices >= 0) & (change_indices < candidate_thresholds_full.size)])
        candidate_thresholds = candidate_thresholds_full[change_indices]
        if candidate_thresholds.size == 0: candidate_thresholds = candidate_thresholds_full[-1:]

        selected_conditions = []
        selected_thresholds = []
        previous_condition = np.zeros_like(s1_slice, dtype=bool)
        threshold_index = 0
        window_volume = moving_kernel_size_2d**2

        while threshold_index < candidate_thresholds.size:
            continue_search = True

            while continue_search and threshold_index < candidate_thresholds.size:
                current_threshold = candidate_thresholds[threshold_index]
                threshold_index += 1

                current_condition = (s1_slice < current_threshold) & segmentation_domain
                condition_difference = current_condition.astype(float) - previous_condition.astype(float)
                local_change_density = np.max(moving_mean_2d(condition_difference, moving_kernel_size_2d))

                if local_change_density > window_density_threshold:
                    selected_conditions.append(current_condition.copy())
                    selected_thresholds.append(current_threshold)
                    previous_condition = current_condition.copy()
                    continue_search = False

                if np.sum(current_condition) >= n_domain_pixels - window_volume:
                    continue_search = False

        if len(selected_conditions) == 0:
            slice_masks.append([])
            selected_thresholds_by_slice.append([])
            continue

        selected_conditions.append(selected_conditions[-1].copy())
        selected_thresholds.append(candidate_thresholds_full[-1])

        material_masks_2d = [(filtered_slice < selected_thresholds[0]) & segmentation_domain]

        for condition_index in range(1, len(selected_conditions)):
            material_masks_2d.append((selected_conditions[condition_index].astype(int) - selected_conditions[condition_index - 1].astype(int)).astype(bool) & segmentation_domain)

        if np.any(mask_container_slice):
            material_masks_2d[-1] = mask_container_slice
            for mask_index in range(len(material_masks_2d) - 1):
                material_masks_2d[mask_index] = 2 * material_masks_2d[mask_index].astype(int) - material_masks_2d[-1].astype(int) == 2
        else:
            material_masks_2d = material_masks_2d[:-1]

        slice_masks.append(material_masks_2d)
        selected_thresholds_by_slice.append(selected_thresholds)

    threshold_keys = set()

    for selected_thresholds in selected_thresholds_by_slice:
        for threshold in selected_thresholds[:-1]:  # exclude final artificial threshold
            threshold_keys.add(int(round(float(threshold) / threshold_step)))

    threshold_keys = sorted(threshold_keys)
    global_thresholds = np.asarray([threshold_key * threshold_step for threshold_key in threshold_keys], dtype=float)

    material_masks = []

    if global_thresholds.size > 0:
        material_masks = [np.zeros(volume_shape, dtype=bool) for _ in range(global_thresholds.size)]

        for slice_index in range(volume_shape[0]):
            s1_slice = s1_tmp[slice_index]
            filtered_slice = filtered_reconstructions[slice_index]
            mask_container_slice = mask_container[slice_index]

            segmentation_domain = ~mask_container_slice if np.any(mask_container_slice) else np.ones_like(s1_slice, dtype=bool)

            global_conditions = [(s1_slice < threshold) & segmentation_domain for threshold in global_thresholds]

            material_masks[0][slice_index, :, :] = (filtered_slice < global_thresholds[0]) & segmentation_domain

            for threshold_index in range(1, global_thresholds.size):
                material_masks[threshold_index][slice_index, :, :] = (
                    global_conditions[threshold_index].astype(int)
                    - global_conditions[threshold_index - 1].astype(int)
                ).astype(bool) & segmentation_domain

    if np.any(mask_container):
        material_masks.append(mask_container.copy())

        for mask_index in range(len(material_masks) - 1):
            material_masks[mask_index] = (
                2 * material_masks[mask_index].astype(int)
                - material_masks[-1].astype(int)
            ) == 2

    return material_masks, selected_thresholds_by_slice