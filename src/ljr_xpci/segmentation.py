import numpy as np
import astra
from scipy.ndimage import shift
from .preprocessing import filter_ft
from .diemoz_filter import apply_single_shot_ei_filter
import numpy as np
from .radon_tools import _astra_iradon_parallel
from .radon_tools_cached import astra_iradon_parallel_cached
from ljr_xpci.utility_functions import err_n
from .utility_functions import select_threshold_conditions_batched, err_n

def _moving_mean_1d(data, window_size, axis):
    data = np.asarray(data, dtype=float); window_size = int(window_size)
    if window_size < 1: raise ValueError("window_size must be positive.")

    moved_data = np.moveaxis(data, axis, 0); n_samples = moved_data.shape[0]
    left_radius = (window_size - 1) // 2; right_radius = window_size // 2
    start_indices = np.maximum(np.arange(n_samples) - left_radius, 0); end_indices = np.minimum(np.arange(n_samples) + right_radius + 1, n_samples)

    cumulative_data = np.concatenate([np.zeros_like(moved_data[:1]), np.cumsum(moved_data, axis=0)], axis=0)
    moving_sum = np.take(cumulative_data, end_indices, axis=0) - np.take(cumulative_data, start_indices, axis=0)
    moving_count = (end_indices - start_indices).reshape((n_samples,) + (1,) * (moved_data.ndim - 1))

    return np.moveaxis(moving_sum / moving_count, 0, axis)


def _moving_mean_2d(data, window_size):
    return _moving_mean_1d(_moving_mean_1d(data, window_size, axis=0), window_size, axis=1)


def _normalise_filter_fraction(filter_fraction_by_axis, ndim):
    if np.isscalar(filter_fraction_by_axis): return tuple(float(filter_fraction_by_axis) for _ in range(ndim))
    filter_fraction_by_axis = tuple(filter_fraction_by_axis)
    if len(filter_fraction_by_axis) != ndim: raise ValueError("filter_fraction_by_axis must be scalar or have one entry for each data dimension.")
    return filter_fraction_by_axis


def ljr_segment(log_projection_noise, detector_pixel_size, gamma_jei, noise_mask, theta_degrees, mask_container=None, projection_shift=0.0, threshold_step=5e-12, filter_fraction_by_axis=0.4, max_mean_window_count=None, n_std=2.0, moving_window_size=7):
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
    max_mean_window_count : float, optional
        Mask-change threshold. If None, set to 0.75*moving_window_size**2.
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

    filter_fraction_by_axis = _normalise_filter_fraction(filter_fraction_by_axis, log_projection_noise.ndim)
    if max_mean_window_count is None: max_mean_window_count = 0.75 * moving_window_size**2
    if mask_container is None: mask_container = np.zeros_like(noise_mask, dtype=bool)
    else: mask_container = np.asarray(mask_container).astype(bool)

    # print("python before filter_FT mean/min/max =", np.shape(log_projection_noise), np.mean(log_projection_noise), np.min(log_projection_noise), np.max(log_projection_noise))
    
    filtered_log_projection = np.real(filter_ft(log_projection_noise, filter_fraction_by_axis))
    # print("python after filter_FT mean/min/max =", np.shape(filtered_log_projection), np.mean(filtered_log_projection), np.min(filtered_log_projection), np.max(filtered_log_projection))
          
    central_slice_index = int(np.ceil(n_slices / 2.0) - 1)

    if test_mat:
        noisy_reconstruction = test_data["s_noise"].astype(float)
        filtered_reconstruction = test_data["s_Ln_1"].astype(float)
    else:
        #noisy_reconstruction = _astra_iradon_parallel(log_projection_noise[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size
        noisy_reconstruction = astra_iradon_parallel_cached(log_projection_noise[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size

        #print("noisy_reconstruction max abs diff =", np.max(np.abs(noisy_reconstruction - noisy_reconstruction_cached)))
        #print("noisy_reconstruction relative error =", err_n(noisy_reconstruction, noisy_reconstruction_cached))
        #print("mean/min/max/std =", np.mean(noisy_reconstruction), np.min(noisy_reconstruction), np.max(noisy_reconstruction), np.std(noisy_reconstruction))
        #print("cached mean/min/max/std =", np.mean(noisy_reconstruction_cached), np.min(noisy_reconstruction_cached), np.max(noisy_reconstruction_cached), np.std(noisy_reconstruction_cached))
        #print("norm =", np.linalg.norm(noisy_reconstruction.ravel()))
        #print("norm diff =", np.linalg.norm((noisy_reconstruction - noisy_reconstruction_cached).ravel()))
        
        
        filtered_reconstruction = astra_iradon_parallel_cached(filtered_log_projection[central_slice_index], theta_degrees, n_detectors) / detector_pixel_size
          
    # print('python mask_noise mean/min/max = ',np.shape(noise_mask), np.mean(noise_mask), np.min(noise_mask), np.max(noise_mask), np.std(noise_mask, ddof=1));
    # print('python s_noise mean/min/max = ',np.shape(noisy_reconstruction), np.mean(noisy_reconstruction), np.min(noisy_reconstruction), np.max(noisy_reconstruction), np.std(noisy_reconstruction, ddof=1));
    
    # print('python s_Ln_1 mean/min/max = ',np.shape(filtered_reconstruction), np.mean(filtered_reconstruction), np.min(filtered_reconstruction), np.max(filtered_reconstruction), np.std(filtered_reconstruction, ddof=1));
      

    masked_noise_reconstruction = noisy_reconstruction * noise_mask

    # print('python s_m before > 0 mean/min/max = ',np.mean(np.fft.fft(masked_noise_reconstruction)), np.mean(masked_noise_reconstruction),np.min(masked_noise_reconstruction), np.max(masked_noise_reconstruction), np.std(masked_noise_reconstruction, ddof=1)); 
    positive_noise_values = masked_noise_reconstruction[masked_noise_reconstruction > 0]
    minimum_threshold = n_std * np.std(positive_noise_values.ravel(), ddof=1) if positive_noise_values.size > 0 else 0.0
 
    
    # print('python s_m after > 0 mean/min/max = ',np.mean(np.fft.fft(positive_noise_values)), np.mean(positive_noise_values),np.min(positive_noise_values), np.max(positive_noise_values), np.std(positive_noise_values, ddof=1)); 

      
    # print("Python d_real_min =", minimum_threshold)

    candidate_thresholds = np.arange(0.0, 1e-8 + 0.5 * threshold_step, threshold_step)
    candidate_thresholds = candidate_thresholds[candidate_thresholds >= minimum_threshold]

    if candidate_thresholds.size == 0: return []

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_noise, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)
          
    # print('python LnT_ss mean/min/max = ', np.mean(single_shot_projection), np.min(single_shot_projection), np.max(single_shot_projection));
          

    single_shot_reconstructions = np.zeros((n_slices, n_detectors, n_detectors), dtype=float)
    for slice_index in range(n_slices):
        shifted_sinogram = shift(single_shot_projection[slice_index], shift=(0.0, projection_shift), order=1, mode="constant", cval=0.0, prefilter=False)
        single_shot_reconstructions[slice_index] = astra_iradon_parallel_cached(shifted_sinogram, theta_degrees, n_detectors) / detector_pixel_size
    
    # print("Python ST size =", single_shot_reconstructions.shape)
    st_mean = np.mean(single_shot_reconstructions, axis=(1, 2)); st_min = np.min(single_shot_reconstructions, axis=(1, 2)); st_max = np.max(single_shot_reconstructions, axis=(1, 2))
    # print("Python ST mean/min/max =", st_mean, st_min, st_max)      
          
    s1_tmp = np.squeeze(np.mean(single_shot_reconstructions, axis=0))
    if test_mat:
        s1_tmp = test_data["s1_tmp"].astype(float) #?

    selected_conditions = []; selected_thresholds = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool)
    threshold_index = 0

    #import time
    #t1 = time.time()
    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            current_threshold = candidate_thresholds[threshold_index]; threshold_index += 1
            current_condition = s1_tmp < current_threshold
            condition_difference = current_condition.astype(float) - previous_condition.astype(float)
            local_change_count = np.max(_moving_mean_2d(condition_difference, moving_window_size)) * moving_window_size**2

            if local_change_count > max_mean_window_count:
                selected_conditions.append(current_condition.copy()); selected_thresholds.append(current_threshold); previous_condition = current_condition.copy(); continue_search = False
            if np.sum(current_condition) >= n_detectors**2 - moving_window_size**2: continue_search = False
                
       #print("segmentation original time =", time.time() - t1)            

    '''t1 = time.time()
    selected_conditions_batched, selected_thresholds_batched = select_threshold_conditions_batched(s1_tmp, candidate_thresholds, moving_window_size, max_mean_window_count, n_detectors, memory_budget_gb=1.0, max_chunk_size=200)

    print("number selected conditions =", len(selected_conditions))
    print("number selected conditions batched =", len(selected_conditions_batched))

    if len(selected_conditions) == len(selected_conditions_batched):
        selected_conditions_array = np.stack(selected_conditions).astype(float) if len(selected_conditions) > 0 else np.empty((0, *s1_tmp.shape))
        selected_conditions_batched_array = np.stack(selected_conditions_batched).astype(float) if len(selected_conditions_batched) > 0 else np.empty((0, *s1_tmp.shape))

        print("selected_conditions max abs diff =", np.max(np.abs(selected_conditions_array - selected_conditions_batched_array)) if selected_conditions_array.size > 0 else 0.0)
        print("selected_conditions relative error =", err_n(selected_conditions_array, selected_conditions_batched_array) if selected_conditions_array.size > 0 else 0.0)
        print("selected_thresholds max abs diff =", np.max(np.abs(np.asarray(selected_thresholds) - selected_thresholds_batched)) if selected_thresholds_batched.size > 0 else 0.0)
        print("selected_thresholds relative error =", err_n(np.asarray(selected_thresholds), selected_thresholds_batched) if selected_thresholds_batched.size > 0 else 0.0)
    else:
        print("selected_conditions relative error = different number of selected conditions")
        print("selected_thresholds =", np.asarray(selected_thresholds))
        print("selected_thresholds_batched =", selected_thresholds_batched)
    print("Segmentation updated time =", time.time() - t1)            
    '''
                
                

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



