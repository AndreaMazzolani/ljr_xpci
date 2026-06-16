import atexit
import numpy as np
import astra


_radon_cuda_context = None
_iradon_cuda_context = None


def _theta_key(theta_degrees):
    theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    return tuple(np.round(theta_degrees, 12))


def _delete_astra_context(context):
    if context is None: return
    if context.get("algorithm_id") is not None: astra.algorithm.delete(context["algorithm_id"])
    if context.get("volume_id") is not None: astra.data2d.delete(context["volume_id"])
    if context.get("sinogram_id") is not None: astra.data2d.delete(context["sinogram_id"])
    if context.get("reconstruction_id") is not None: astra.data2d.delete(context["reconstruction_id"])


def clear_cached_astra_objects():
    global _radon_cuda_context, _iradon_cuda_context
    _delete_astra_context(_radon_cuda_context); _delete_astra_context(_iradon_cuda_context)
    _radon_cuda_context = None; _iradon_cuda_context = None


atexit.register(clear_cached_astra_objects)


def _get_cached_radon_cuda_context(number_pixels, detector_count, theta_degrees, gpu_index=0):
    global _radon_cuda_context
    theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    context_key = ("radon_cuda", int(number_pixels), int(detector_count), _theta_key(theta_degrees), int(gpu_index))

    if _radon_cuda_context is not None and _radon_cuda_context["context_key"] == context_key: return _radon_cuda_context
    _delete_astra_context(_radon_cuda_context)

    theta_radians = np.deg2rad(theta_degrees)
    volume_geometry = astra.create_vol_geom(number_pixels, number_pixels)
    projection_geometry = astra.create_proj_geom("parallel", 1.0, detector_count, theta_radians)
    volume_id = astra.data2d.create("-vol", volume_geometry, 0)
    sinogram_id = astra.data2d.create("-sino", projection_geometry, 0)

    cfg = astra.astra_dict("FP_CUDA"); cfg["VolumeDataId"] = volume_id; cfg["ProjectionDataId"] = sinogram_id; cfg["option"] = {"GPUindex": int(gpu_index)}
    algorithm_id = astra.algorithm.create(cfg)

    _radon_cuda_context = {"context_key": context_key, "volume_id": volume_id, "sinogram_id": sinogram_id, "reconstruction_id": None, "algorithm_id": algorithm_id, "sinogram_shape": (theta_degrees.size, detector_count)}
    return _radon_cuda_context


def _get_cached_iradon_cuda_context(output_size, detector_count, theta_degrees, filter_type="ram-lak", gpu_index=0):
    global _iradon_cuda_context
    theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    context_key = ("iradon_cuda", int(output_size), int(detector_count), _theta_key(theta_degrees), filter_type, int(gpu_index))

    if _iradon_cuda_context is not None and _iradon_cuda_context["context_key"] == context_key: return _iradon_cuda_context
    _delete_astra_context(_iradon_cuda_context)

    theta_radians = np.deg2rad(theta_degrees)
    projection_geometry = astra.create_proj_geom("parallel", 1.0, detector_count, theta_radians)
    volume_geometry = astra.create_vol_geom(output_size, output_size)
    sinogram_id = astra.data2d.create("-sino", projection_geometry, 0)
    reconstruction_id = astra.data2d.create("-vol", volume_geometry, 0)

    cfg = astra.astra_dict("FBP_CUDA"); cfg["ProjectionDataId"] = sinogram_id; cfg["ReconstructionDataId"] = reconstruction_id; cfg["option"] = {"FilterType": filter_type, "GPUindex": int(gpu_index)}
    algorithm_id = astra.algorithm.create(cfg)

    _iradon_cuda_context = {"context_key": context_key, "volume_id": None, "sinogram_id": sinogram_id, "reconstruction_id": reconstruction_id, "algorithm_id": algorithm_id, "reconstruction_shape": (output_size, output_size)}
    return _iradon_cuda_context


def astra_radon_parallel_cached(image, theta_degrees, detector_count=None, gpu_index=0):
    """
    Compute one parallel-beam projection using cached ASTRA CUDA objects.

    image : np.ndarray, shape (Nx, Nx)
        Input 2D image.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    detector_count : int, optional
        Number of detector samples.
    gpu_index : int, optional
        GPU index used by ASTRA.
    """

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.shape[0] != image.shape[1]: raise ValueError("image must have shape (Nx, Nx).")
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA is not available in this Python environment.")

    number_pixels = image.shape[0]; detector_count = number_pixels if detector_count is None else int(detector_count)
    context = _get_cached_radon_cuda_context(number_pixels, detector_count, theta_degrees, gpu_index=gpu_index)

    astra.data2d.store(context["volume_id"], image)
    astra.data2d.store(context["sinogram_id"], np.zeros(context["sinogram_shape"], dtype=np.float32))
    astra.algorithm.run(context["algorithm_id"])

    return astra.data2d.get(context["sinogram_id"])


def astra_radon_parallel_many_cached(images, theta_degrees, detector_count=None, gpu_index=0):
    """
    Compute several parallel-beam projections using cached ASTRA CUDA objects.

    images : np.ndarray, shape (Nv, Nx, Nx)
        Stack of input 2D images.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    detector_count : int, optional
        Number of detector samples.
    gpu_index : int, optional
        GPU index used by ASTRA.
    """

    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1] != images.shape[2]: raise ValueError("images must have shape (Nv, Nx, Nx).")
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA is not available in this Python environment.")

    number_images, number_pixels, _ = images.shape; detector_count = number_pixels if detector_count is None else int(detector_count)
    context = _get_cached_radon_cuda_context(number_pixels, detector_count, theta_degrees, gpu_index=gpu_index)

    sinograms = np.empty((number_images, *context["sinogram_shape"]), dtype=np.float32)
    zero_sinogram = np.zeros(context["sinogram_shape"], dtype=np.float32)

    for image_index in range(number_images):
        astra.data2d.store(context["volume_id"], images[image_index])
        astra.data2d.store(context["sinogram_id"], zero_sinogram)
        astra.algorithm.run(context["algorithm_id"])
        sinograms[image_index] = astra.data2d.get(context["sinogram_id"])

    return sinograms


def astra_iradon_parallel_cached(sinogram, theta_degrees, output_size, filter_type="ram-lak", gpu_index=0):
    """
    Reconstruct one 2D slice from a parallel-beam sinogram using cached ASTRA CUDA objects.

    sinogram : np.ndarray, shape (Ntheta, Nx)
        Parallel-beam sinogram.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    output_size : int
        Reconstructed image size.
    filter_type : str
        FBP filter name.
    gpu_index : int, optional
        GPU index used by ASTRA.
    """

    sinogram = np.asarray(sinogram, dtype=np.float32)
    if sinogram.ndim != 2: raise ValueError("sinogram must have shape (Ntheta, Nx).")
    if sinogram.shape[0] != len(theta_degrees): raise ValueError("sinogram.shape[0] must match len(theta_degrees).")
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA is not available in this Python environment.")

    context = _get_cached_iradon_cuda_context(output_size, sinogram.shape[1], theta_degrees, filter_type=filter_type, gpu_index=gpu_index)

    astra.data2d.store(context["sinogram_id"], sinogram)
    astra.data2d.store(context["reconstruction_id"], np.zeros(context["reconstruction_shape"], dtype=np.float32))
    astra.algorithm.run(context["algorithm_id"])

    return astra.data2d.get(context["reconstruction_id"])


def ljr_thickness_cached(material_masks, pixel_size, theta_degrees, theta_indices=None, gpu_index=0):
    """
    Compute material thickness sinograms and detector-axis derivatives using cached ASTRA CUDA projections.

    material_masks : list[np.ndarray]
        Material masks, each with shape (Nx, Nx).
    pixel_size : float
        Pixel size used to convert projected mask counts into physical thickness.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    theta_indices : np.ndarray, optional
        Projection-angle indices to keep.
    gpu_index : int, optional
        GPU index used by ASTRA.

    Returns
    -------
    thickness_sinograms : np.ndarray, shape (Nv, Ntheta, Nx)
        Material thickness sinograms.
    thickness_derivatives : np.ndarray, shape (Nv, Ntheta, Nx)
        Detector-axis finite-difference derivatives.
    """

    number_materials = len(material_masks); theta_degrees = np.asarray(theta_degrees, dtype=float)
    if number_materials == 0: raise ValueError("material_masks must contain at least one mask.")

    theta_indices = np.arange(theta_degrees.size) if theta_indices is None else np.asarray(theta_indices, dtype=int)
    number_pixels = np.asarray(material_masks[0]).shape[0]
    mask_stack = np.empty((number_materials, number_pixels, number_pixels), dtype=np.float32)

    for material_index, current_mask in enumerate(material_masks):
        current_mask = np.asarray(current_mask, dtype=np.float32)
        if current_mask.shape != (number_pixels, number_pixels): raise ValueError("All masks must have shape (Nx, Nx).")
        mask_stack[material_index] = current_mask

    thickness_sinograms = astra_radon_parallel_many_cached(mask_stack, theta_degrees, detector_count=number_pixels, gpu_index=gpu_index) * pixel_size
    thickness_sinograms = thickness_sinograms[:, theta_indices, :]

    thickness_derivatives = thickness_sinograms.copy()
    thickness_derivatives[:, :, 1:-1] = (thickness_sinograms[:, :, 2:] - thickness_sinograms[:, :, :-2]) / (2.0 * pixel_size)

    return thickness_sinograms.astype(float), thickness_derivatives.astype(float)