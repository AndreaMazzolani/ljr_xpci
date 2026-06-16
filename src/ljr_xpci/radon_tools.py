import numpy as np
import astra


def astra_radon_parallel(image, theta_degrees, detector_count=None, projector_type="linear"):
    """
    Compute a parallel-beam projection using ASTRA.

    Parameters
    ----------
    image : np.ndarray, shape (Nx, Nx)
        Input 2D image.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    detector_count : int, optional
        Number of detector samples. If omitted, detector_count = Nx.
    projector_type : str, optional
        ASTRA 2D projector type.

    Returns
    -------
    sinogram : np.ndarray, shape (Ntheta, detector_count)
        Parallel-beam sinogram.
    """

    image = np.asarray(image, dtype=np.float32); theta_radians = np.deg2rad(np.asarray(theta_degrees, dtype=float))
    if image.ndim != 2 or image.shape[0] != image.shape[1]: raise ValueError("image must have shape (Nx, Nx).")
    detector_count = image.shape[0] if detector_count is None else int(detector_count)

    volume_geometry = astra.create_vol_geom(image.shape[0], image.shape[1])
    projection_geometry = astra.create_proj_geom("parallel", 1.0, detector_count, theta_radians)
    projector_id = astra.create_projector(projector_type, projection_geometry, volume_geometry)
    volume_id = astra.data2d.create("-vol", volume_geometry, image)
    sinogram_id = astra.data2d.create("-sino", projection_geometry, 0)

    algorithm_id = None
    try:
        cfg = astra.astra_dict("FP"); cfg["ProjectorId"] = projector_id; cfg["VolumeDataId"] = volume_id; cfg["ProjectionDataId"] = sinogram_id
        algorithm_id = astra.algorithm.create(cfg); astra.algorithm.run(algorithm_id)
        sinogram = astra.data2d.get(sinogram_id)
    finally:
        if algorithm_id is not None: astra.algorithm.delete(algorithm_id)
        astra.data2d.delete(volume_id); astra.data2d.delete(sinogram_id); astra.projector.delete(projector_id)

    return sinogram

def _astra_iradon_parallel(sinogram, theta_degrees, output_size, filter_type="ram-lak", projector_type="linear"):
    """
    Reconstruct one 2D slice from a parallel-beam sinogram.

    Parameters
    ----------
    sinogram : np.ndarray, shape (Ntheta, Nx)
        Parallel-beam log-projection sinogram. Axis 0 is projection angle, axis 1 is detector x.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    output_size : int
        Number of pixels in each reconstructed image dimension.
    filter_type : str
        FBP filter name.
    projector_type : str
        ASTRA 2D projector type.

    Returns
    -------
    reconstruction : np.ndarray, shape (output_size, output_size)
        Reconstructed image.
    """

    sinogram = np.asarray(sinogram, dtype=np.float32); theta_radians = np.deg2rad(np.asarray(theta_degrees, dtype=float))
    if sinogram.ndim != 2: raise ValueError("sinogram must have shape (Ntheta, Nx).")
    if sinogram.shape[0] != theta_radians.size: raise ValueError("sinogram.shape[0] must match len(theta_degrees).")

    projection_geometry = astra.create_proj_geom("parallel", 1.0, sinogram.shape[1], theta_radians)
    volume_geometry = astra.create_vol_geom(output_size, output_size)
    projector_id = astra.create_projector(projector_type, projection_geometry, volume_geometry)
    sinogram_id = astra.data2d.create("-sino", projection_geometry, sinogram)
    reconstruction_id = astra.data2d.create("-vol", volume_geometry, 0)

    algorithm_id = None
    try:
        cfg = astra.astra_dict("FBP"); cfg["ProjectorId"] = projector_id; cfg["ProjectionDataId"] = sinogram_id; cfg["ReconstructionDataId"] = reconstruction_id; cfg["option"] = {"FilterType": filter_type}
        algorithm_id = astra.algorithm.create(cfg); astra.algorithm.run(algorithm_id)
        reconstruction = astra.data2d.get(reconstruction_id)
    finally:
        if algorithm_id is not None: astra.algorithm.delete(algorithm_id)
        astra.data2d.delete(sinogram_id); astra.data2d.delete(reconstruction_id); astra.projector.delete(projector_id)

    return reconstruction



def ljr_thickness(material_masks, pixel_size, theta_degrees, theta_indices=None):
    """
    Compute material thickness sinograms and their detector-axis derivatives.

    Parameters
    ----------
    material_masks : list[np.ndarray]
        List of material masks. Each mask has shape (Nx, Nx).
    pixel_size : float
        Pixel size used to convert projected mask counts into physical thickness.
    theta_degrees : np.ndarray, shape (Ntheta,)
        Projection angles in degrees.
    theta_indices : np.ndarray, optional
        Indices of the projection angles to keep. If omitted, all angles are used.

    Returns
    -------
    thickness_sinograms : np.ndarray, shape (Nv, Ntheta, Nx)
        Material thickness sinograms. Nv is the number of material masks.
    thickness_derivatives : np.ndarray, shape (Nv, Ntheta, Nx)
        Central finite-difference derivative of the thickness sinograms along the detector axis.
    """

    number_materials = len(material_masks); theta_degrees = np.asarray(theta_degrees, dtype=float)
    if number_materials == 0: raise ValueError("material_masks must contain at least one mask.")
    theta_indices = np.arange(theta_degrees.size) if theta_indices is None else np.asarray(theta_indices, dtype=int)

    number_pixels = np.asarray(material_masks[0]).shape[0]; number_angles = theta_indices.size
    thickness_sinograms = np.zeros((number_materials, number_angles, number_pixels), dtype=float); thickness_derivatives = np.zeros_like(thickness_sinograms)

    for material_index, current_mask in enumerate(material_masks):
        current_mask = np.asarray(current_mask, dtype=np.float32)
        if current_mask.shape != (number_pixels, number_pixels): raise ValueError("All masks must have shape (Nx, Nx).")

        thickness_tmp = astra_radon_parallel(current_mask, theta_degrees, detector_count=number_pixels) * pixel_size
        thickness_tmp = thickness_tmp[theta_indices, :]  # shape (Ntheta, Nx)

        derivative_tmp = thickness_tmp.copy()
        derivative_tmp[:, 1:-1] = (thickness_tmp[:, 2:] - thickness_tmp[:, :-2]) / (2.0 * pixel_size)

        thickness_sinograms[material_index] = thickness_tmp
        thickness_derivatives[material_index] = derivative_tmp
        

    return thickness_sinograms, thickness_derivatives




