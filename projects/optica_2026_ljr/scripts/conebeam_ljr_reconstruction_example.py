"""
Run the parallel-beam Linear Joint Reconstruction (LJR) pipeline.

This script loads a projection stack, arranges it as (Ns, Ntheta, Nx), estimates 3D material masks,
computes material thickness sinograms, fits one common set of beta/delta coefficients across the selected slices,
and reconstructs refraction-like and absorption-like CT volumes using parallel-beam ASTRA reconstruction.
"""

import sys
import argparse
import time
from pathlib import Path

from scipy.io import savemat  # saving final reconstructions in MATLAB-compatible .mat format
import numpy as np
import tifffile as tiff

from ljr_xpci.segmentation import ljr_segment_3D,ljr_segment_conebeam, LJR_conebeam_ind, ljr_segment_conebeam_bisection
from ljr_xpci.radon_tools import ljr_thickness_3D, ljr_thickness_conebeam_3D, astra_iradon_parallel_3D, astra_iradon_conebeam_3D, astra_radon_parallel_many, astra_radon_conebeam_3D
from ljr_xpci.ljr_solver import estimate_material_coefficients_3D, compose_ljr_projection_components_3D
from ljr_xpci.utility_functions import err_n
from ljr_xpci.diemoz_filter import apply_single_shot_ei_filter

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"  # project-specific config folder
sys.path.insert(0, str(CONFIG_DIR))  # allows importing local_paths.py from the config folder

from local_paths import get_default_data_root  # returns the default local data folder for the current machine

def main():
    project_folder = Path(__file__).resolve().parents[1]  # projects/optica_2026_ljr folder
    sys.path.insert(0, str(project_folder / "configs"))  # repeated config-path insertion, can be removed later
    parser = argparse.ArgumentParser(description="Run the LJR pipeline on the Khush dataset.")  # command-line parser
    parser.add_argument("--data-root", type=Path, default=None, help="Path to the folder containing the project data.")  # optional external data root
    args = parser.parse_args()  # read command-line inputs

    data_root = args.data_root if args.data_root is not None else get_default_data_root()  # use user path or machine default
    if data_root is None: raise RuntimeError("Data root was not provided and this computer was not recognized. Use --data-root /path/to/data.")  # stop if no data location is available

    data_folder = data_root / "pd" / "Khush_folder"  # private input data folder
    data_file = data_folder / "DATA.tif"  # raw projection stack

    output_folder = project_folder / "outputs" / "conebeam_ljr_test"  # public project output folder
    mat_folder = output_folder / "mat_files"
    mat_folder.mkdir(parents=True, exist_ok=True)
    data = tiff.imread(data_file).astype(float)  # load raw projection stack as float
    data[np.isinf(data)] = 0  # remove infinite entries from the raw data

    ang_step = 1#
    slice_indices = np.arange(0, 9)  # selected slice indices, Python indexing
    n_silces = len(slice_indices)

    source_origin_distance = 1.085
    source_detector_distance = 1.403
    threshold_step = 5e-12# 5e-10
    filter_fraction_by_axis = (None, None, 0.4)
    n_std = 2.7
    moving_window_size = 5
    moving_kernel_size_3d = (moving_window_size, moving_window_size, 1)
    window_density_threshold = 0.75

    k_mean2 = 8.8e10
    detector_pixel_size = 6e-6
    gamma_jei = -21111744.110502686 * 0.2e-11
    j_ei = -21000.0

    data = tiff.imread(data_file).astype(float)
    data[np.isinf(data)] = 0

    n_slices_total, number_x_pixels, number_angles = data.shape
    theta_degrees = np.linspace(0.0, 360.0, number_angles + 1)[:-1]

    log_projection = data[slice_indices, :, :]
    log_projection = np.transpose(log_projection, (0, 2, 1))
    log_projection[np.isinf(log_projection)] = 0

    n_detectors = log_projection.shape[-1]
    pix_x = np.arange(1, n_detectors + 1, dtype=float)[:, None]
    pix_y = np.arange(1, n_detectors + 1, dtype=float)[None, :]

    noise_cx, noise_cy = 1300.0, 1200.0
    noise_mask = (pix_x - noise_cx) ** 2 + (pix_y - noise_cy) ** 2 > 1100.0 ** 2

    container_ext_cx, container_ext_cy = 1338.0, 1286.5
    container_int_cx, container_int_cy = 1331.0, 1268.0
    mask_container_ext = (pix_x - container_ext_cx) ** 2 + (pix_y - container_ext_cy) ** 2 < 887.0 ** 2
    mask_container_int = (pix_x - container_int_cx) ** 2 + (pix_y - container_int_cy) ** 2 < 745.0 ** 2
    mask_container = mask_container_ext & ~mask_container_int

    log_projection_scaled = log_projection[:, ::ang_step, ::ang_step] / (2.0 * k_mean2)
    log_projection_unscaled = log_projection[:, ::ang_step, ::ang_step]
    theta_degrees_used = theta_degrees[::ang_step]
    noise_mask_used = noise_mask[::ang_step, ::ang_step]
    mask_container_used = mask_container[::ang_step, ::ang_step]
    mask_container_3D = np.broadcast_to(mask_container_used[None, :, :], (log_projection_scaled.shape[0], *mask_container_used.shape)).copy()

    
    central_slice_index = log_projection_scaled.shape[0] // 2
    n_slices, n_angles, n_detectors = log_projection_scaled.shape

    cone_geometry = {}
    cone_geometry["source_origin_distance"] = 1.085
    cone_geometry["source_detector_distance"] = 1.403
    cone_geometry["origin_detector_distance"] = cone_geometry["source_detector_distance"] - cone_geometry["source_origin_distance"]
    cone_geometry["volume_shape"] = (n_slices, n_detectors, n_detectors)
    cone_geometry["n_detector_y"] = n_slices
    cone_geometry["n_detector_x"] = n_detectors
    cone_geometry["detector_pixel_size_x"] = detector_pixel_size
    cone_geometry["detector_pixel_size_y"] = detector_pixel_size

    _, single_shot_complex_projection = apply_single_shot_ei_filter(log_projection_scaled, gamma_jei, detector_pixel_size)
    single_shot_projection = np.real(single_shot_complex_projection)

    single_projections_transp = np.transpose(single_shot_projection, (1, 2, 0))
    s_parallel_raw_3D = astra_iradon_parallel_3D(single_projections_transp, theta_degrees_used, output_size=n_detectors) / detector_pixel_size
    s_conebeam_raw_3D = astra_iradon_conebeam_3D(single_shot_projection, theta_degrees_used, cone_geometry) / detector_pixel_size

    s_parallel_raw = s_parallel_raw_3D[:, :, central_slice_index]
    s_conebeam_raw = s_conebeam_raw_3D[central_slice_index]
    
    '''
    print("parallel raw 3D shape =", s_parallel_raw_3D.shape)
    print("conebeam raw 3D shape =", s_conebeam_raw_3D.shape)
    print("parallel central shape =", s_parallel_raw.shape)
    print("conebeam central shape =", s_conebeam_raw.shape)
    print("err cone/parallel =", err_n(s_parallel_raw, s_conebeam_raw))
    print("err cone.T/parallel =", err_n(s_parallel_raw, s_conebeam_raw.T))
    print("err flipud(cone)/parallel =", err_n(s_parallel_raw, np.flipud(s_conebeam_raw)))
    print("err fliplr(cone)/parallel =", err_n(s_parallel_raw, np.fliplr(s_conebeam_raw)))    
    '''
    
    sl = int(log_projection.shape[0]); ii = slice_indices[0]; mw = int(moving_window_size); kz = int(moving_kernel_size_3d[2]); wd = int(round(window_density_threshold * 100)); sd = int(round(n_std * 100)); ts = int(round(threshold_step * 1e12)); ff = int(round(filter_fraction_by_axis[-1] * 100)); ag = int(ang_step)
    mat_file_3D = output_folder / f"CB2tmp_recs_sl_{sl}_ii_{ii}_mw_{mw}_kz_{kz}_wd_{wd}_sd_{sd}_ts_{ts}_ff_{ff}_ag_{ag}.mat"
    savemat(mat_file_3D, {"single_projections_transp": single_projections_transp, "single_shot_projection": single_shot_projection, "s_conebeam_raw_3D": s_conebeam_raw_3D, "s_parallel_raw_3D": s_parallel_raw_3D, "ang_step":ang_step})
    print("saved mat file:", mat_file_3D)
    
    
    print("Parallel LJR reconstruction")

    t0 = time.time()
    material_masks_3D, selected_thresholds_3D = ljr_segment_3D(
        log_projection_scaled,
        detector_pixel_size,
        gamma_jei,
        theta_degrees_used,
        noise_mask_used,
        mask_container=mask_container_3D,
        threshold_step=threshold_step,
        filter_fraction_by_axis=filter_fraction_by_axis,
        n_std=n_std,
        moving_kernel_size_3d=moving_kernel_size_3d,
        window_density_threshold=window_density_threshold,
    )
    print("segmentation time =", time.time() - t0)
    print("number of segmented materials =", len(material_masks_3D))
 
    print("Conebeam LJR reconstruction")

    noise_mask_conebeam = np.flip(noise_mask_used, axis=-2)
    mask_container_conebeam = np.flip(mask_container_3D, axis=-2)
    
    t0 = time.time()
    material_masks_conebeam, selected_thresholds_conebeam = ljr_segment_conebeam(
        log_projection_scaled,
        detector_pixel_size,
        gamma_jei,
        theta_degrees_used,
        noise_mask_used,
        cone_geometry,
        mask_container=mask_container_3D,
        # projection_shift=projection_shift,
        threshold_step=threshold_step,
        filter_fraction_by_axis=filter_fraction_by_axis,
        window_density_threshold=window_density_threshold,
        n_std=n_std,
        moving_kernel_size_3d=moving_kernel_size_3d,
    )
    print("conebeam segmentation time =", time.time() - t0)
    print("number of conebeam material masks =", len(material_masks_conebeam))
     
    
    moving_kernel_size_2d = int(moving_kernel_size_3d[0])

    t0 = time.time()
    material_masks_conebeam_ind, selected_thresholds_conebeam_ind = ljr_segment_conebeam_bisection(
        log_projection_scaled,
        detector_pixel_size,
        gamma_jei,
        theta_degrees_used,
        noise_mask_used,
        cone_geometry,
        mask_container=mask_container_3D,
        threshold_step=threshold_step,
        filter_fraction_by_axis=filter_fraction_by_axis,
        window_density_threshold=window_density_threshold,
        n_std=n_std,
        moving_kernel_size_2d=moving_kernel_size_2d, 
    )
                                         
    print("LJR_conebeam_ind time =", time.time() - t0)

    
    t0 = time.time()
    thickness_sinograms_3D, thickness_derivatives_3D = ljr_thickness_3D(material_masks_3D, detector_pixel_size, theta_degrees_used)
    print("thickness time =", time.time() - t0)

    t0 = time.time()
    thickness_sinograms_conebeam_3D, thickness_derivatives_conebeam_3D = ljr_thickness_3D(material_masks_conebeam, detector_pixel_size, theta_degrees_used)
    print("thickness conebeam time =", time.time() - t0)

    t0 = time.time()
    thickness_sinograms_conebeam_ind_3D, thickness_derivatives_conebeam_ind_3D = ljr_thickness_3D(material_masks_conebeam_ind, detector_pixel_size, theta_degrees_used)
    print("thickness conebeam ind time=", time.time() - t0)

    mean_thickness_3D = np.mean(thickness_sinograms_3D, axis=(1, 2, 3))
    mean_thickness_conebeam = np.mean(thickness_sinograms_conebeam_3D, axis=(1, 2, 3))
    mean_thickness_conebeam_ind = np.mean(thickness_sinograms_conebeam_ind_3D, axis=(1, 2, 3))

    mean_derivatives_3D = np.mean(np.abs(thickness_derivatives_3D), axis=(1, 2, 3))
    mean_derivatives_conebeam = np.mean(np.abs(thickness_derivatives_conebeam_3D), axis=(1, 2, 3))
    mean_derivatives_conebeam_ind = np.mean(np.abs(thickness_derivatives_conebeam_ind_3D), axis=(1, 2, 3))
    
    coefficient_estimates_3D = estimate_material_coefficients_3D(-log_projection_unscaled, thickness_sinograms_3D, thickness_derivatives_3D, j_ei, k_mean2)
 
    coefficient_estimates_conebeam = estimate_material_coefficients_3D(-log_projection_unscaled, thickness_sinograms_conebeam_3D, thickness_derivatives_conebeam_3D, j_ei, k_mean2)
    
    coefficient_estimates_conebeam_ind = estimate_material_coefficients_3D(-log_projection_unscaled, thickness_sinograms_conebeam_ind_3D, thickness_derivatives_conebeam_ind_3D, j_ei, k_mean2)
    
     
    projection_refraction_integral_3D, projection_absorption_3D = compose_ljr_projection_components_3D(coefficient_estimates_3D, thickness_sinograms_3D)

    projection_refraction_integral_conebeam, projection_absorption_conebeam = compose_ljr_projection_components_3D(coefficient_estimates_conebeam, thickness_sinograms_conebeam_3D)

    projection_refraction_integral_conebeam_ind, projection_absorption_conebeam_ind = compose_ljr_projection_components_3D(coefficient_estimates_conebeam_ind, thickness_sinograms_conebeam_ind_3D)

    print("err projection refraction conebeam =", err_n(projection_refraction_integral_3D, projection_refraction_integral_conebeam))
    print("err projection absorption conebeam =", err_n(projection_absorption_3D, projection_absorption_conebeam))

    print("err projection refraction conebeam ind =", err_n(projection_refraction_integral_3D, projection_refraction_integral_conebeam_ind))
    print("err projection absorption conebeam ind =", err_n(projection_absorption_3D, projection_absorption_conebeam_ind))

    t0 = time.time()
    s_ref_est_3D = astra_iradon_parallel_3D(projection_refraction_integral_3D, theta_degrees_used) / detector_pixel_size
    s_abs_est_3D = astra_iradon_parallel_3D(projection_absorption_3D, theta_degrees_used) / detector_pixel_size
    print("final reconstruction time =", time.time() - t0)

    print("projection_refraction_integral_conebeam shape =", projection_refraction_integral_conebeam.shape)
    print("projection_absorption_conebeam shape =", projection_absorption_conebeam.shape)
    print("projection_refraction_integral_conebeam_ind shape =", projection_refraction_integral_conebeam_ind.shape)
    print("projection_absorption_conebeam_ind shape =", projection_absorption_conebeam_ind.shape)

    def as_conebeam_projection(projection):
        projection = np.asarray(projection)
        if projection.ndim == 2:
            return projection[None, :, :]              # (1, Ntheta, Nx)
        if projection.ndim == 3:
            return np.transpose(projection, (2, 0, 1)) # (Ns, Ntheta, Nx)
        raise ValueError(f"Unexpected projection shape: {projection.shape}")

    s_ref_est_cb_stack = astra_iradon_conebeam_3D(as_conebeam_projection(projection_refraction_integral_conebeam), theta_degrees_used, cone_geometry) / detector_pixel_size
    s_abs_est_cb_stack = astra_iradon_conebeam_3D(as_conebeam_projection(projection_absorption_conebeam), theta_degrees_used, cone_geometry) / detector_pixel_size
    s_ref_est_cb_ind_stack = astra_iradon_conebeam_3D(as_conebeam_projection(projection_refraction_integral_conebeam_ind), theta_degrees_used, cone_geometry) / detector_pixel_size
    s_abs_est_cb_ind_stack = astra_iradon_conebeam_3D(as_conebeam_projection(projection_absorption_conebeam_ind), theta_degrees_used, cone_geometry) / detector_pixel_size
    
    def as_volume_stack(volume):
        volume = np.asarray(volume)
        if volume.ndim == 2:
            return volume[:, :, None]
        if volume.ndim == 3:
            return volume
        raise ValueError(f"Unexpected volume shape: {volume.shape}")
    
    s_ref_est_cb_ind = np.transpose(s_ref_est_cb_ind_stack, (1, 2, 0))
    s_abs_est_cb_ind = np.transpose(s_abs_est_cb_ind_stack, (1, 2, 0))
    s_ref_est_cb = np.transpose(s_ref_est_cb_stack, (1, 2, 0))
    s_abs_est_cb = np.transpose(s_abs_est_cb_stack, (1, 2, 0))

    s_ref_est_3D_cmp = as_volume_stack(s_ref_est_3D)
    s_abs_est_3D_cmp = as_volume_stack(s_abs_est_3D)
    s_ref_est_cb_cmp = as_volume_stack(s_ref_est_cb)
    s_abs_est_cb_cmp = as_volume_stack(s_abs_est_cb)
    s_ref_est_cb_ind_cmp = as_volume_stack(s_ref_est_cb_ind)
    s_abs_est_cb_ind_cmp = as_volume_stack(s_abs_est_cb_ind)

    print("s_ref_est_3D_cmp shape =", s_ref_est_3D_cmp.shape)
    print("s_ref_est_cb_cmp shape =", s_ref_est_cb_cmp.shape)
    print("s_ref_est_cb_ind_cmp shape =", s_ref_est_cb_ind_cmp.shape)

    print("err final ref/abs cb =", err_n(s_ref_est_3D_cmp, s_ref_est_cb_cmp), err_n(s_abs_est_3D_cmp, s_abs_est_cb_cmp))
    print("err final ref/abs ind cb =", err_n(s_ref_est_3D_cmp, s_ref_est_cb_ind_cmp), err_n(s_abs_est_3D_cmp, s_abs_est_cb_ind_cmp))

    print("err projection refraction old cb vs ind cb =", err_n(projection_refraction_integral_conebeam, projection_refraction_integral_conebeam_ind))
    print("err projection absorption old cb vs ind cb =", err_n(projection_absorption_conebeam, projection_absorption_conebeam_ind))

    print("err final ref old cb vs ind cb =", err_n(s_ref_est_cb, s_ref_est_cb_ind))
    print("err final abs old cb vs ind cb =", err_n(s_abs_est_cb, s_abs_est_cb_ind))

    print("number of old conebeam masks =", len(material_masks_conebeam))
    print("number of ind conebeam masks =", len(material_masks_conebeam_ind))
                                         
    sl = int(log_projection.shape[0]); ii = slice_indices[0]; mw = int(moving_window_size); kz = int(moving_kernel_size_3d[2]); wd = int(round(window_density_threshold * 100)); sd = int(round(n_std * 100)); ts = int(round(threshold_step * 1e12)); ff = int(round(filter_fraction_by_axis[-1] * 100)); ag = int(ang_step)
    mat_file_3D = output_folder / f"CB_recs_sl_{sl}_ii_{ii}_mw_{mw}_kz_{kz}_wd_{wd}_sd_{sd}_ts_{ts}_ff_{ff}_ag_{ag}.mat"
    savemat(mat_file_3D, {"s_ref_est_3D": s_ref_est_3D, "s_abs_est_3D": s_abs_est_3D, "s_ref_est_cb": s_ref_est_cb, "s_abs_est_cb": s_abs_est_cb, "s_ref_est_cb_ind": s_ref_est_cb_ind, "s_abs_est_cb_ind": s_abs_est_cb_ind, "projection_refraction_integral_3D": projection_refraction_integral_3D, "projection_refraction_integral_conebeam": projection_refraction_integral_conebeam, "projection_absorption_3D": projection_absorption_3D, "projection_absorption_conebeam": projection_absorption_conebeam, "coefficient_estimates_3D": coefficient_estimates_3D, "coefficient_estimates_conebeam": coefficient_estimates_conebeam, "ang_step": ang_step})
    print("saved mat file:", mat_file_3D) 
    

    return s_ref_est_3D, s_abs_est_3D, coefficient_estimates_3D


if __name__ == "__main__":
    main()
    
    
    
    
    