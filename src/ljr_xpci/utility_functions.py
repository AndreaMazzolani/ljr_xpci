import numpy as np


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


def choose_threshold_chunk_size(image_shape, memory_budget_gb=1.0, max_chunk_size=200, dtype=np.float32):
    number_pixels = int(np.prod(image_shape)); bytes_per_threshold = number_pixels * (1 + 5 * np.dtype(dtype).itemsize)
    return max(1, min(int(max_chunk_size), int(memory_budget_gb * 1024**3 // bytes_per_threshold)))


def _moving_sum_along_axis(data, window_size, axis):
    n = data.shape[axis]; left = window_size // 2; right = window_size - left - 1
    start_indices = np.maximum(np.arange(n) - left, 0); end_indices = np.minimum(np.arange(n) + right + 1, n)
    zero_shape = list(data.shape); zero_shape[axis] = 1
    cumulative_data = np.concatenate([np.zeros(zero_shape, dtype=data.dtype), np.cumsum(data, axis=axis)], axis=axis)
    return np.take(cumulative_data, end_indices, axis=axis) - np.take(cumulative_data, start_indices, axis=axis)


def moving_mean_2d_stack(data_stack, window_size):
    data_stack = np.asarray(data_stack)
    row_counts = _moving_sum_along_axis(np.ones(data_stack.shape[-2:], dtype=data_stack.dtype), window_size, axis=0)
    local_counts = _moving_sum_along_axis(row_counts, window_size, axis=1)
    local_sums = _moving_sum_along_axis(_moving_sum_along_axis(data_stack, window_size, axis=-2), window_size, axis=-1)
    return local_sums / local_counts[None, :, :]


def select_threshold_conditions_batched(s1_tmp, candidate_thresholds, moving_window_size, max_mean_window_count, n_detectors, memory_budget_gb=1.0, max_chunk_size=200):
    selected_conditions_batched = []; selected_thresholds_batched = []
    previous_condition = np.zeros_like(s1_tmp, dtype=bool)
    threshold_index = 0
    threshold_chunk_size = choose_threshold_chunk_size(s1_tmp.shape, memory_budget_gb=memory_budget_gb, max_chunk_size=max_chunk_size, dtype=np.float32)

    while threshold_index < candidate_thresholds.size:
        continue_search = True
        while continue_search and threshold_index < candidate_thresholds.size:
            chunk_start = threshold_index; chunk_stop = min(threshold_index + threshold_chunk_size, candidate_thresholds.size)
            threshold_chunk = candidate_thresholds[chunk_start:chunk_stop]

            current_conditions_chunk = s1_tmp[None, :, :] < threshold_chunk[:, None, None]
            condition_differences_chunk = current_conditions_chunk.astype(np.float32) - previous_condition[None, :, :].astype(np.float32)

            local_change_counts = np.max(moving_mean_2d_stack(condition_differences_chunk, moving_window_size), axis=(1, 2)) * moving_window_size**2
            current_pixel_counts = np.sum(current_conditions_chunk, axis=(1, 2))

            event_positions = np.flatnonzero((local_change_counts > max_mean_window_count) | (current_pixel_counts >= n_detectors**2 - moving_window_size**2))

            if event_positions.size == 0:
                threshold_index = chunk_stop
            else:
                chunk_event_index = int(event_positions[0])
                current_threshold = threshold_chunk[chunk_event_index]
                current_condition = current_conditions_chunk[chunk_event_index]
                threshold_index = chunk_start + chunk_event_index + 1

                if local_change_counts[chunk_event_index] > max_mean_window_count:
                    selected_conditions_batched.append(current_condition.copy()); selected_thresholds_batched.append(current_threshold); previous_condition = current_condition.copy()

                continue_search = False

    return selected_conditions_batched, np.asarray(selected_thresholds_batched)




