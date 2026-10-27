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

from ljr_xpci.segmentation import ljr_segment_3D
from ljr_xpci.radon_tools import ljr_thickness_3D, astra_iradon_parallel_3D
from ljr_xpci.ljr_solver import estimate_material_coefficients_3D, compose_ljr_projection_components_3D

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

    output_folder = project_folder / "outputs" / "parallel_ljr_test"  # public project output folder
    mat_folder = output_folder / "mat_files"
    mat_folder.mkdir(parents=True, exist_ok=True)
    data = tiff.imread(data_file).astype(float)  # load raw projection stack as float
    data[np.isinf(data)] = 0  # remove infinite entries from the raw data

    ang_step = 1
    slice_indices = np.arange(3, 4)  # selected slice indices, Python indexing
    #slice_indices = np.arange(3, 6)  # selected slice indices, Python indexing

    threshold_step = 5e-12
    filter_fraction_by_axis = (None, None, 0.4)
    n_std = 2.7
    moving_window_size = 5
    moving_kernel_size_3d = (moving_window_size, moving_window_size, 3)
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

    t0 = time.time()
    thickness_sinograms_3D, thickness_derivatives_3D = ljr_thickness_3D(material_masks_3D, detector_pixel_size, theta_degrees_used)
    print("thickness time =", time.time() - t0)

    t0 = time.time()
    coefficient_estimates_3D = estimate_material_coefficients_3D(-log_projection_unscaled, thickness_sinograms_3D, thickness_derivatives_3D, j_ei, k_mean2)
    print("coefficient estimation time =", time.time() - t0)

    t0 = time.time()
    projection_refraction_integral_3D, projection_absorption_3D = compose_ljr_projection_components_3D(coefficient_estimates_3D, thickness_sinograms_3D)
    print("projection composition time =", time.time() - t0)

    t0 = time.time()
    s_ref_est_3D = astra_iradon_parallel_3D(projection_refraction_integral_3D, theta_degrees_used) / detector_pixel_size
    s_abs_est_3D = astra_iradon_parallel_3D(projection_absorption_3D, theta_degrees_used) / detector_pixel_size
    print("final reconstruction time =", time.time() - t0)

    
    sl = int(log_projection.shape[0]); ii = slice_indices[0]; mw = int(moving_window_size); kz = int(moving_kernel_size_3d[2]); wd = int(round(window_density_threshold * 100)); sd = int(round(n_std * 100)); ts = int(round(threshold_step * 1e12)); ff = int(round(filter_fraction_by_axis[-1] * 100)); ag = int(ang_step)
    mat_file_3D = output_folder / f"SP_recs_sl_{sl}_ii_{ii}_mw_{mw}_kz_{kz}_wd_{wd}_sd_{sd}_ts_{ts}_ff_{ff}_ag_{ag}.mat"
    savemat(mat_file_3D, {"s_ref_est_3D": s_ref_est_3D, "s_abs_est_3D": s_abs_est_3D, "ang_step":ang_step})
    
    print("saved mat file:", mat_file_3D)

    return s_ref_est_3D, s_abs_est_3D, coefficient_estimates_3D


if __name__ == "__main__":
    main()
    
    
    
    
    