import atexit
import numpy as np
import astra


_radon_cuda_context = None
_iradon_cuda_context = None
_conebeam_radon_cuda_context = None
_conebeam_iradon_cuda_context = None


def _theta_key(theta_degrees):
    theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    return tuple(np.round(theta_degrees, 12))

def _delete_astra_context_3D(context):
    if context is None: return
    if context.get("algorithm_id") is not None: astra.algorithm.delete(context["algorithm_id"])
    if context.get("volume_id") is not None: astra.data3d.delete(context["volume_id"])
    if context.get("projection_id") is not None: astra.data3d.delete(context["projection_id"])
    if context.get("reconstruction_id") is not None: astra.data3d.delete(context["reconstruction_id"])

def _delete_astra_context(context):
    if context is None: return
    if context.get("algorithm_id") is not None: astra.algorithm.delete(context["algorithm_id"])
    if context.get("volume_id") is not None: astra.data2d.delete(context["volume_id"])
    if context.get("sinogram_id") is not None: astra.data2d.delete(context["sinogram_id"])
    if context.get("reconstruction_id") is not None: astra.data2d.delete(context["reconstruction_id"])


def clear_astra_objects():
    global _radon_cuda_context, _iradon_cuda_context
    _delete_astra_context(_radon_cuda_context); _delete_astra_context(_iradon_cuda_context)
    _radon_cuda_context = None; _iradon_cuda_context = None

def clear_cached_conebeam_astra_objects():
    global _conebeam_radon_cuda_context, _conebeam_iradon_cuda_context
    _delete_astra_context_3D(_conebeam_radon_cuda_context); _delete_astra_context_3D(_conebeam_iradon_cuda_context)
    _conebeam_radon_cuda_context = None; _conebeam_iradon_cuda_context = None


atexit.register(clear_cached_conebeam_astra_objects)

atexit.register(clear_astra_objects)


def _get_radon_cuda_context(number_pixels, detector_count, theta_degrees, gpu_index=0):
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


def _get_iradon_cuda_context(output_size, detector_count, theta_degrees, filter_type="ram-lak", gpu_index=0):
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

def _cone_geometry_values(theta_degrees, cone_geometry):
    theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    volume_shape = tuple(int(v) for v in cone_geometry["volume_shape"])  # (Nz, Ny, Nx)
    n_detector_x = int(cone_geometry["n_detector_x"]); n_detector_y = int(cone_geometry["n_detector_y"])
    voxel_size = float(cone_geometry.get("voxel_size", 1.0))
    detector_spacing_x = float(cone_geometry["detector_pixel_size_x"]) / voxel_size; detector_spacing_y = float(cone_geometry["detector_pixel_size_y"]) / voxel_size
    source_origin_distance = float(cone_geometry["source_origin_distance"]) / voxel_size; origin_detector_distance = float(cone_geometry["origin_detector_distance"]) / voxel_size
    return theta_degrees, volume_shape, n_detector_x, n_detector_y, detector_spacing_x, detector_spacing_y, source_origin_distance, origin_detector_distance

def _get_cached_iradon_conebeam_context(theta_degrees, cone_geometry, filter_type="ram-lak", gpu_index=0):
    global _conebeam_iradon_cuda_context
    theta_degrees, volume_shape, n_detector_x, n_detector_y, detector_spacing_x, detector_spacing_y, source_origin_distance, origin_detector_distance = _cone_geometry_values(theta_degrees, cone_geometry)
    context_key = ("iradon_cone_3D", volume_shape, n_detector_x, n_detector_y, detector_spacing_x, detector_spacing_y, source_origin_distance, origin_detector_distance, _theta_key(theta_degrees), filter_type, int(gpu_index))
    if _conebeam_iradon_cuda_context is not None and _conebeam_iradon_cuda_context["context_key"] == context_key: return _conebeam_iradon_cuda_context
    _delete_astra_context_3D(_conebeam_iradon_cuda_context)

    n_z, n_y, n_x = volume_shape
    projection_geometry = astra.create_proj_geom("cone", detector_spacing_x, detector_spacing_y, n_detector_y, n_detector_x, np.deg2rad(theta_degrees), source_origin_distance, origin_detector_distance)
    volume_geometry = astra.create_vol_geom(n_y, n_x, n_z)
    projection_id = astra.data3d.create("-sino", projection_geometry, 0)
    reconstruction_id = astra.data3d.create("-vol", volume_geometry, 0)

    cfg = astra.astra_dict("FDK_CUDA"); cfg["ProjectionDataId"] = projection_id; cfg["ReconstructionDataId"] = reconstruction_id; cfg["option"] = {"FilterType": filter_type, "GPUindex": int(gpu_index)}
    algorithm_id = astra.algorithm.create(cfg)

    _conebeam_iradon_cuda_context = {"context_key": context_key, "volume_id": None, "projection_id": projection_id, "reconstruction_id": reconstruction_id, "algorithm_id": algorithm_id, "projection_shape": (n_detector_y, theta_degrees.size, n_detector_x), "volume_shape": volume_shape}
    return _conebeam_iradon_cuda_context

def astra_iradon_conebeam_3D(projection, theta_degrees, cone_geometry, filter_type="ram-lak", gpu_index=0):
    """
    Reconstruct a 3D cone-beam volume using ASTRA FDK_CUDA.

    Parameters
    ----------
    projection : np.ndarray, shape (Ndet_y, Ntheta, Ndet_x)
        Cone-beam projection stack. Axis 0 is detector y, axis 1 is angle, axis 2 is detector x.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    cone_geometry : dict
        Geometry dictionary. Required: source_origin_distance and either source_detector_distance or origin_detector_distance. Optional: detector_pixel_size_x, detector_pixel_size_y, n_detector_x, n_detector_y, volume_shape, reconstruction_voxel_size.
    filter_type : str
        FDK filter name.
    gpu_index : int
        GPU index used by ASTRA.

    Returns
    -------
    reconstruction : np.ndarray, shape (Ndet_y, Ndet_x, Ndet_x) by default
        Reconstructed volume. The central slice reconstruction[projection.shape[0]//2] has shape (Ndet_x, Ndet_x), matching astra_iradon_parallel output size.
    """

    projection = np.asarray(projection, dtype=np.float32); theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    if projection.ndim != 3: raise ValueError("projection must have shape (Ndet_y, Ntheta, Ndet_x).")
    if projection.shape[1] != theta_degrees.size: raise ValueError("projection.shape[1] must match len(theta_degrees).")
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA is not available.")

    n_detector_y_data, n_angles, n_detector_x_data = projection.shape
    n_detector_x = int(cone_geometry.get("n_detector_x", n_detector_x_data)); n_detector_y = int(cone_geometry.get("n_detector_y", n_detector_y_data))
    if projection.shape != (n_detector_y, theta_degrees.size, n_detector_x): raise ValueError("projection shape does not match cone_geometry detector size.")

    source_origin_distance_m = float(cone_geometry["source_origin_distance"])
    if "source_detector_distance" in cone_geometry:
        source_detector_distance_m = float(cone_geometry["source_detector_distance"]); origin_detector_distance_m = source_detector_distance_m - source_origin_distance_m
    else:
        origin_detector_distance_m = float(cone_geometry["origin_detector_distance"]); source_detector_distance_m = source_origin_distance_m + origin_detector_distance_m
    if source_origin_distance_m <= 0 or origin_detector_distance_m <= 0: raise ValueError("source_origin_distance and origin_detector_distance must be positive.")

    magnification = source_detector_distance_m / source_origin_distance_m
    detector_pixel_size_x = float(cone_geometry.get("detector_pixel_size_x", cone_geometry.get("detector_pixel_size", 1.0)))
    detector_pixel_size_y = float(cone_geometry.get("detector_pixel_size_y", detector_pixel_size_x))
    voxel_size = float(cone_geometry.get("reconstruction_voxel_size", detector_pixel_size_x / magnification))

    detector_spacing_x = detector_pixel_size_x / voxel_size; detector_spacing_y = detector_pixel_size_y / voxel_size
    source_origin_distance = source_origin_distance_m / voxel_size; origin_detector_distance = origin_detector_distance_m / voxel_size

    volume_shape = tuple(int(v) for v in cone_geometry.get("volume_shape", (n_detector_y, n_detector_x, n_detector_x)))  # (Nz, Ny, Nx)
    n_z, n_y, n_x = volume_shape

    projection_geometry = astra.create_proj_geom("cone", detector_spacing_x, detector_spacing_y, n_detector_y, n_detector_x, np.deg2rad(theta_degrees), source_origin_distance, origin_detector_distance)
    volume_geometry = astra.create_vol_geom(n_y, n_x, n_z)

    projection_id = astra.data3d.create("-sino", projection_geometry, projection)
    reconstruction_id = astra.data3d.create("-vol", volume_geometry, 0)
    cfg = astra.astra_dict("FDK_CUDA"); cfg["ProjectionDataId"] = projection_id; cfg["ReconstructionDataId"] = reconstruction_id; cfg["option"] = {"FilterType": filter_type, "GPUindex": int(gpu_index)}
    algorithm_id = astra.algorithm.create(cfg)

    try:
        astra.algorithm.run(algorithm_id)
        reconstruction = astra.data3d.get(reconstruction_id)
    finally:
        astra.algorithm.delete(algorithm_id); astra.data3d.delete(projection_id); astra.data3d.delete(reconstruction_id)

    return reconstruction[:, ::-1, :]

def _get_cached_radon_conebeam_context(theta_degrees, cone_geometry, gpu_index=0):
    global _conebeam_radon_cuda_context
    theta_degrees, volume_shape, n_detector_x, n_detector_y, detector_spacing_x, detector_spacing_y, source_origin_distance, origin_detector_distance = _cone_geometry_values(theta_degrees, cone_geometry)
    context_key = ("radon_cone_3D", volume_shape, n_detector_x, n_detector_y, detector_spacing_x, detector_spacing_y, source_origin_distance, origin_detector_distance, _theta_key(theta_degrees), int(gpu_index))
    if _conebeam_radon_cuda_context is not None and _conebeam_radon_cuda_context["context_key"] == context_key: return _conebeam_radon_cuda_context
    _delete_astra_context_3D(_conebeam_radon_cuda_context)

    n_z, n_y, n_x = volume_shape
    projection_geometry = astra.create_proj_geom("cone", detector_spacing_x, detector_spacing_y, n_detector_y, n_detector_x, np.deg2rad(theta_degrees), source_origin_distance, origin_detector_distance)
    volume_geometry = astra.create_vol_geom(n_y, n_x, n_z)
    volume_id = astra.data3d.create("-vol", volume_geometry, 0)
    projection_id = astra.data3d.create("-sino", projection_geometry, 0)

    cfg = astra.astra_dict("FP3D_CUDA"); cfg["VolumeDataId"] = volume_id; cfg["ProjectionDataId"] = projection_id; cfg["option"] = {"GPUindex": int(gpu_index)}
    algorithm_id = astra.algorithm.create(cfg)

    _conebeam_radon_cuda_context = {"context_key": context_key, "volume_id": volume_id, "projection_id": projection_id, "reconstruction_id": None, "algorithm_id": algorithm_id, "projection_shape": (n_detector_y, theta_degrees.size, n_detector_x), "volume_shape": volume_shape}
    return _conebeam_radon_cuda_context

'''
def astra_radon_conebeam_3D(volume, theta_degrees, cone_geometry, gpu_index=0):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3: raise ValueError("volume must have shape (Nz, Ny, Nx).")
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA is not available in this Python environment.")
    context = _get_cached_radon_conebeam_context(theta_degrees, cone_geometry, gpu_index=gpu_index)
    if volume.shape != context["volume_shape"]: raise ValueError(f"volume must have shape {context['volume_shape']}.")

    astra.data3d.store(context["volume_id"], volume)
    astra.data3d.store(context["projection_id"], np.zeros(context["projection_shape"], dtype=np.float32))
    astra.algorithm.run(context["algorithm_id"])

    return astra.data3d.get(context["projection_id"])
'''

def astra_radon_conebeam_3D(volume, theta_degrees, cone_geometry, gpu_index=0):
    volume = np.asarray(volume, dtype=np.float32); theta_degrees = np.asarray(theta_degrees, dtype=float).reshape(-1)
    if volume.ndim != 3: raise ValueError("volume must have shape (Nz, Ny, Nx).")
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA is not available.")

    theta_degrees, volume_shape, n_detector_x, n_detector_y, detector_spacing_x, detector_spacing_y, source_origin_distance, origin_detector_distance = _cone_geometry_values(theta_degrees, cone_geometry)
    n_z, n_y, n_x = volume_shape
    if volume.shape != volume_shape: raise ValueError(f"volume must have shape {volume_shape}.")

    projection_geometry = astra.create_proj_geom("cone", detector_spacing_x, detector_spacing_y, n_detector_y, n_detector_x, np.deg2rad(theta_degrees), source_origin_distance, origin_detector_distance)
    volume_geometry = astra.create_vol_geom(n_y, n_x, n_z)

    volume_id = astra.data3d.create("-vol", volume_geometry, volume)
    projection_id = astra.data3d.create("-sino", projection_geometry, np.zeros((n_detector_y, theta_degrees.size, n_detector_x), dtype=np.float32))

    cfg = astra.astra_dict("FP3D_CUDA"); cfg["VolumeDataId"] = volume_id; cfg["ProjectionDataId"] = projection_id; cfg["option"] = {"GPUindex": int(gpu_index)}
    algorithm_id = astra.algorithm.create(cfg); astra.algorithm.run(algorithm_id)

    projection = astra.data3d.get(projection_id)

    astra.algorithm.delete(algorithm_id); astra.data3d.delete(volume_id); astra.data3d.delete(projection_id)

    if not np.all(np.isfinite(projection)): raise RuntimeError("Conebeam forward projection contains non-finite values.")
    if np.min(projection) < -1e-6: raise RuntimeError(f"Conebeam forward projection contains negative values: min={np.min(projection)}")

    return projection

'''
def ljr_thickness_conebeam_3D(material_masks, pixel_size, theta_degrees, cone_geometry, gpu_index=0):
    number_materials = len(material_masks)
    if number_materials == 0: raise ValueError("material_masks must contain at least one mask.")

    thickness_sinograms = []
    for current_mask in material_masks:
        current_mask = np.asarray(current_mask, dtype=np.float32)
        thickness_sinograms.append(astra_radon_conebeam_3D(current_mask, theta_degrees, cone_geometry, gpu_index=gpu_index) * pixel_size)

    thickness_sinograms = np.asarray(thickness_sinograms, dtype=float)  # shape (Nv, Ndet_y, Ntheta, Ndet_x)
    thickness_derivatives = thickness_sinograms.copy()
    thickness_derivatives[..., 1:-1] = (thickness_sinograms[..., 2:] - thickness_sinograms[..., :-2]) / (2.0 * pixel_size)

    return thickness_sinograms, thickness_derivatives
'''

def astra_radon_parallel(image, theta_degrees, detector_count=None, gpu_index=0):
    """
    Compute one parallel-beam projection using ASTRA CUDA objects.

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
    context = _get_radon_cuda_context(number_pixels, detector_count, theta_degrees, gpu_index=gpu_index)

    astra.data2d.store(context["volume_id"], image)
    astra.data2d.store(context["sinogram_id"], np.zeros(context["sinogram_shape"], dtype=np.float32))
    astra.algorithm.run(context["algorithm_id"])

    return astra.data2d.get(context["sinogram_id"])


def astra_radon_parallel_many(images, theta_degrees, detector_count=None, gpu_index=0):
    """
    Compute several parallel-beam projections using ASTRA CUDA objects.

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
    context = _get_radon_cuda_context(number_pixels, detector_count, theta_degrees, gpu_index=gpu_index)

    sinograms = np.empty((number_images, *context["sinogram_shape"]), dtype=np.float32)
    zero_sinogram = np.zeros(context["sinogram_shape"], dtype=np.float32)

    for image_index in range(number_images):
        astra.data2d.store(context["volume_id"], images[image_index])
        astra.data2d.store(context["sinogram_id"], zero_sinogram)
        astra.algorithm.run(context["algorithm_id"])
        sinograms[image_index] = astra.data2d.get(context["sinogram_id"])

    return sinograms


def astra_iradon_parallel(sinogram, theta_degrees, output_size, filter_type="ram-lak", gpu_index=0):
    """
    Reconstruct one 2D slice from a parallel-beam sinogram using ASTRA CUDA objects.

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

    context = _get_iradon_cuda_context(output_size, sinogram.shape[1], theta_degrees, filter_type=filter_type, gpu_index=gpu_index)

    astra.data2d.store(context["sinogram_id"], sinogram)
    astra.data2d.store(context["reconstruction_id"], np.zeros(context["reconstruction_shape"], dtype=np.float32))
    astra.algorithm.run(context["algorithm_id"])

    return astra.data2d.get(context["reconstruction_id"])

def astra_iradon_parallel_3D(projection_3D, theta_degrees, output_size=None, filter_type="ram-lak", gpu_index=0):
    """
    Reconstruct multiple parallel-beam slices.

    projection_3D : np.ndarray, shape (Ntheta, Nx, Ns)
        Projection stack with slice index on the last axis.
    """

    projection_3D = np.asarray(projection_3D, dtype=np.float32)
    if projection_3D.ndim == 2: return astra_iradon_parallel(projection_3D, theta_degrees, projection_3D.shape[-1], filter_type=filter_type, gpu_index=gpu_index)
    if projection_3D.ndim != 3: raise ValueError("projection_3D must have shape (Ntheta, Nx, Ns).")
    if projection_3D.shape[0] != len(theta_degrees): raise ValueError("projection_3D.shape[0] must match len(theta_degrees).")

    n_angles, n_detectors, n_slices = projection_3D.shape
    output_size = n_detectors if output_size is None else int(output_size)

    reconstruction_3D = np.empty((output_size, output_size, n_slices), dtype=float)
    for slice_index in range(n_slices):
        reconstruction_3D[:, :, slice_index] = astra_iradon_parallel(projection_3D[:, :, slice_index], theta_degrees, output_size, filter_type=filter_type, gpu_index=gpu_index)

    return reconstruction_3D

def ljr_thickness(material_masks, pixel_size, theta_degrees, theta_indices=None, gpu_index=0):
    """
    Compute material thickness sinograms and detector-axis derivatives using ASTRA CUDA projections.

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

    thickness_sinograms = astra_radon_parallel_many(mask_stack, theta_degrees, detector_count=number_pixels, gpu_index=gpu_index) * pixel_size
    thickness_sinograms = thickness_sinograms[:, theta_indices, :]

    thickness_derivatives = thickness_sinograms.copy()
    thickness_derivatives[:, :, 1:-1] = (thickness_sinograms[:, :, 2:] - thickness_sinograms[:, :, :-2]) / (2.0 * pixel_size)

    return thickness_sinograms.astype(float), thickness_derivatives.astype(float)


def ljr_thickness_3D(material_masks_3D, pixel_size, theta_degrees, theta_indices=None, gpu_index=0):
    """
    Compute 3D material thickness sinograms and detector-axis derivatives.

    material_masks_3D : list[np.ndarray]
        Material masks, each with shape (Ns, Nx, Nx).
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
    thickness_sinograms_3D : np.ndarray, shape (Nv, Ntheta, Nx, Ns)
        Material thickness sinograms.
    thickness_derivatives_3D : np.ndarray, shape (Nv, Ntheta, Nx, Ns)
        Detector-axis finite-difference derivatives.
    """

    number_materials = len(material_masks_3D); theta_degrees = np.asarray(theta_degrees, dtype=float)
    if number_materials == 0: raise ValueError("material_masks_3D must contain at least one mask.")

    theta_indices = np.arange(theta_degrees.size) if theta_indices is None else np.asarray(theta_indices, dtype=int)
    first_mask = np.asarray(material_masks_3D[0]); n_slices, number_pixels, number_pixels_y = first_mask.shape
    if number_pixels != number_pixels_y: raise ValueError("Each 3D mask must have shape (Ns, Nx, Nx).")

    mask_stack = np.empty((number_materials, n_slices, number_pixels, number_pixels), dtype=np.float32)
    for material_index, current_mask in enumerate(material_masks_3D):
        current_mask = np.asarray(current_mask, dtype=np.float32)
        if current_mask.shape != (n_slices, number_pixels, number_pixels): raise ValueError("All masks must have shape (Ns, Nx, Nx).")
        mask_stack[material_index] = current_mask

    mask_stack_flat = mask_stack.reshape(number_materials * n_slices, number_pixels, number_pixels)
    tmp = astra_radon_parallel_many(mask_stack_flat, theta_degrees, detector_count=number_pixels, gpu_index=gpu_index) * pixel_size
    thickness_flat = tmp

    thickness_sinograms_3D = thickness_flat.reshape(number_materials, n_slices, theta_degrees.size, number_pixels)
    thickness_sinograms_3D = np.transpose(thickness_sinograms_3D, (0, 2, 3, 1))[:, theta_indices, :, :]

    thickness_derivatives_3D = thickness_sinograms_3D.copy()
    thickness_derivatives_3D[:, :, 1:-1, :] = (thickness_sinograms_3D[:, :, 2:, :] - thickness_sinograms_3D[:, :, :-2, :]) / (2.0 * pixel_size)

    return thickness_sinograms_3D.astype(float), thickness_derivatives_3D.astype(float)



def ljr_thickness_conebeam_3D(material_masks_3D, pixel_size, theta_degrees, cone_geometry, theta_indices=None, gpu_index=0):
    """
    Compute conebeam 3D material thickness sinograms and detector-x derivatives.

    material_masks_3D : list[np.ndarray]
        Material masks, each with shape (Ns, Nx, Nx).
    pixel_size : float
        Pixel size used to convert projected mask counts into physical thickness.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    cone_geometry : dict
        Cone-beam geometry used by ASTRA.
    theta_indices : np.ndarray, optional
        Projection-angle indices to keep.
    gpu_index : int, optional
        GPU index used by ASTRA.

    Returns
    -------
    thickness_sinograms_3D : np.ndarray, shape (Nv, Ntheta, Nx, Ns)
        Material thickness sinograms.
    thickness_derivatives_3D : np.ndarray, shape (Nv, Ntheta, Nx, Ns)
        Detector-x finite-difference derivatives.
    """

    number_materials = len(material_masks_3D); theta_degrees = np.asarray(theta_degrees, dtype=float)
    if number_materials == 0: raise ValueError("material_masks_3D must contain at least one mask.")

    theta_indices = np.arange(theta_degrees.size) if theta_indices is None else np.asarray(theta_indices, dtype=int)
    volume_shape = tuple(int(v) for v in cone_geometry["volume_shape"])
    n_slices, number_pixels_y, number_pixels_x = volume_shape
    if number_pixels_y != number_pixels_x: raise ValueError("cone_geometry['volume_shape'] must be (Ns, Nx, Nx).")

    thickness_sinograms = np.empty((number_materials, n_slices, theta_degrees.size, number_pixels_x), dtype=np.float32)

    for material_index, current_mask in enumerate(material_masks_3D):
        current_mask = np.asarray(current_mask, dtype=np.float32)
        if current_mask.shape != volume_shape: raise ValueError("All masks must match cone_geometry['volume_shape'].")
        
        tmp = astra_radon_conebeam_3D(current_mask, theta_degrees, cone_geometry, gpu_index=gpu_index) * pixel_size
        #tmp = astra_radon_conebeam_3D(current_mask[:, ::-1, :], theta_degrees, cone_geometry, gpu_index=gpu_index) * pixel_size
        if tmp.shape != thickness_sinograms.shape[1:]:
            tmp = np.transpose(tmp, (2, 1, 0))
        thickness_sinograms[material_index] = tmp
    
    thickness_sinograms_3D = np.transpose(thickness_sinograms, (0, 2, 3, 1))[:, theta_indices, :, :]
    #thickness_sinograms_3D = np.transpose(thickness_sinograms, (0, 3, 2, 1))[:, theta_indices, :, :]
    
 
    thickness_derivatives_3D = thickness_sinograms_3D.copy()
    thickness_derivatives_3D[:, :, 1:-1, :] = (thickness_sinograms_3D[:, :, 2:, :] - thickness_sinograms_3D[:, :, :-2, :]) / (2.0 * pixel_size)

    return thickness_sinograms_3D.astype(float), thickness_derivatives_3D.astype(float)


