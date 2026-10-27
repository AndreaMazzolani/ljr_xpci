import argparse  # command-line argument parsing
"""
Run the LJR reconstruction pipeline for the Khush single-shot EI dataset.

This script loads the projection stack, arranges it as (Ns, Ntheta, Nx), and runs the Linear Joint Reconstruction (LJR) workflow described in the manuscript. First, a scaled version of the normalized log projection L is used only for segmentation, where the sample is divided into material masks B_j. These masks define a piecewise-material basis for the object.

The masks B_j are then forward-projected with ASTRA to estimate the material thickness sinograms T_j(theta, x) and their detector-axis derivatives dT_j/dx. The unscaled projection data are then used in the LJR linear least-squares problem to estimate the material coefficients beta_j and delta_j. Finally, the beta-weighted and delta-weighted projection components are reconstructed by inverse Radon transform to obtain absorption-like and refraction-like CT slices.

Data convention
---------------
log_projection : shape (Ns, Ntheta, Nx)
    Ns is the number of selected slices, Ntheta the projection-angle axis, Nx the detector axis.

Main outputs
------------
s_ref_est : shape (Nx, Nx)
    Reconstructed delta-like/refraction map.

s_abs_est : shape (Nx, Nx)
    Reconstructed beta-like/absorption map.
"""

import sys  # used to add the local config folder to the import path
from pathlib import Path  # path handling independent of operating system

import numpy as np  # numerical arrays and vectorised operations
import tifffile as tiff  # reading the projection stack saved as .tif
import matplotlib.pyplot as plt  # plotting and saving final reconstructed slices
from scipy.io import savemat  # saving final reconstructions in MATLAB-compatible .mat format

from ljr_xpci.segmentation import ljr_segment, ljr_segment_3D, ljr_segment_conebeam  
from ljr_xpci.plotting import show_image  # image display helper, currently unused in this script
from ljr_xpci.ljr_solver import estimate_material_coefficients, estimate_material_coefficients_3D, compose_ljr_projection_components, compose_ljr_projection_components_3D  # solves for beta_j, delta_j 
from ljr_xpci.utility_functions import err_n, moving_mean_3d, moving_mean_2d
from ljr_xpci.radon_tools import ljr_thickness, ljr_thickness_3D, ljr_thickness_conebeam_3D, astra_iradon_parallel, astra_iradon_parallel_3D, astra_iradon_conebeam_3D  # computes T_j and reconstructs final delta/beta maps
import time  # timing of the main pipeline stages

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"  # project-specific config folder
sys.path.insert(0, str(CONFIG_DIR))  # allows importing local_paths.py from the config folder

from local_paths import get_default_data_root  # returns the default local data folder for the current machine


def main():  # run the Khush LJR example pipeline

    """
    Execute the full example: load data, build masks B_j, compute thickness terms T_j, estimate beta_j and delta_j, reconstruct final CT slices, and save PNG/MAT outputs.
    """
    
    ang_step = 5  # angular and detector subsampling factor for a faster test run
    slice_indices = np.arange(4, 5)  # selected slice indices, Python indexing
    slice_indices = np.arange(3, 6)  # selected slice indices, Python indexing
    #lice_indices = np.arange(2, 7)  # selected slice indices, Python indexing
    origin_detector_distance = 1.403
    source_origin_distance = 1.085
    threshold_step = 7e-10
    filter_fraction_by_axis=(None, None, 0.4)
    n_std = 2.7
    n_std2 = 2.7
    gam_sc1 = 1 # 0.5
    gam_sc2 = 1 # 2
    moving_window_size = 5
    moving_kernel_size_3d = (moving_window_size,moving_window_size,3)
    window_density_threshold= 0.75

    project_folder = Path(__file__).resolve().parents[1]  # projects/optica_2026_ljr folder
    sys.path.insert(0, str(project_folder / "configs"))  # repeated config-path insertion, can be removed later
    parser = argparse.ArgumentParser(description="Run the LJR pipeline on the Khush dataset.")  # command-line parser
    parser.add_argument("--data-root", type=Path, default=None, help="Path to the folder containing the project data.")  # optional external data root
    args = parser.parse_args()  # read command-line inputs

    data_root = args.data_root if args.data_root is not None else get_default_data_root()  # use user path or machine default
    if data_root is None: raise RuntimeError("Data root was not provided and this computer was not recognized. Use --data-root /path/to/data.")  # stop if no data location is available

    sample_folder = data_root / "pd" / "Khush_folder"  # folder containing the Khush dataset
    data_file = sample_folder / "DATA.tif"  # raw projection stack
    results_file = sample_folder / "khush_results2.npz"  # unused output filename placeholder

    data = tiff.imread(data_file).astype(float)  # load raw projection stack as float
    data[np.isinf(data)] = 0  # remove infinite entries from the raw data

    n_slices_total, number_x_pixels, number_angles = data.shape  # raw data shape: slices, detector x, angles

    k_mean2 = 8.8e10  # effective wavenumber-related scaling used in the LJR model
    detector_pixel_size = 6e-6  # detector pixel size in metres
    theta_degrees = np.linspace(0.0, 360.0, number_angles + 1)[:-1]  # projection angles over 360 degrees
    gamma_jei = -21111744.110502686 * 0.2e-11  # gamma parameter used by the Diemoz-style single-shot filter
    j_ei = -21000.0  # J_EI coefficient multiplying the refraction derivative term in LJR

    N_slice = slice_indices.size  # number of selected slices

    log_projection = data[slice_indices, :, :]  # selected raw log projection slices, shape (Ns, Nx, Ntheta)
    log_projection = np.transpose(log_projection, (0, 2, 1))  # rearrange to LJR convention, shape (Ns, Ntheta, Nx)
    log_projection[np.isinf(log_projection)] = 0  # remove infinite entries after slicing/transposition

    n_detectors = log_projection.shape[-1]  # number of detector pixels Nx

    pix_x = np.arange(1, n_detectors + 1, dtype=float)[:, None]  # detector x grid as column
    pix_y = np.arange(1, n_detectors + 1, dtype=float)[None, :]  # detector y grid as row

    c_x, c_y = 1300.0, 1200.0  # centre of the external noise/background region
    noise_mask = (pix_x - c_x)**2 + (pix_y - c_y)**2 > 1100.0**2  # background mask used to estimate the noise threshold

    c_x2, c_y2 = 1338.0, 1286.5  # centre of the external container mask
    mask_container_ext = (pix_x - c_x2)**2 + (pix_y - c_y2)**2 < 887.0**2  # external container mask, currently unused

    c_x3, c_y3 = 1331.0, 1268.0  # centre of the internal container mask
    mask_container_int = (pix_x - c_x3)**2 + (pix_y - c_y3)**2 < 745.0**2  # internal container mask, currently unused

    mask_container  = mask_container_ext & ~mask_container_int
    log_projection_test = log_projection[:, ::ang_step, ::ang_step] / (2.0 * k_mean2)  # scaled L used only for segmentation, shape (Ns, Ntheta_test, Nx_test)
    log_projection_test_unscaled = log_projection[:, ::ang_step, ::ang_step]  # unscaled L used in the LJR coefficient estimation
    theta_degrees_test = theta_degrees[::ang_step]  # subsampled projection angles
    noise_mask_test = noise_mask[::ang_step, ::ang_step]  # subsampled noise mask matching Nx_test
    mask_container_test = mask_container[::ang_step, ::ang_step]
    central_slice_index = int(np.ceil(log_projection_test.shape[0] / 2.0) - 1)
    log_projection_test_central = log_projection_test[central_slice_index:central_slice_index + 1, :, :]  # shape (1, Ntheta, Nx)

    
    cone_geometry = {
    "source_origin_distance": source_origin_distance,
    "origin_detector_distance": origin_detector_distance,
    "detector_pixel_size_x": detector_pixel_size,
    "detector_pixel_size_y": detector_pixel_size,
    "n_detector_x": log_projection_test.shape[-1],
    "n_detector_y": log_projection_test.shape[0],
    "volume_shape": (log_projection_test.shape[0], log_projection_test.shape[-1], log_projection_test.shape[-1]),
    "voxel_size": detector_pixel_size
    }
    
    
    #s_parallel_central = astra_iradon_parallel(log_projection_test[central_slice_index], theta_degrees_test, log_projection_test.shape[-1]) / detector_pixel_size

    # s_conebeam_central = astra_iradon_conebeam_3D(log_projection_test, theta_degrees_test, cone_geometry) / detector_pixel_size    
    #s_conebeam_3D = astra_iradon_conebeam_3D(log_projection_test, theta_degrees_test, cone_geometry) / detector_pixel_size
    #s_conebeam_central = s_conebeam_3D[central_slice_index]

    #print("recon err_n parallel/conebeam central =", err_n(s_parallel_central, s_conebeam_central))
    
    #print("parallel mean/min/max/std =", np.mean(s_parallel_central), np.min(s_parallel_central), np.max(s_parallel_central), np.std(s_parallel_central, ddof=1))
    #print("conebeam mean/min/max/std =", np.mean(s_conebeam_central), np.min(s_conebeam_central), np.max(s_conebeam_central), np.std(s_conebeam_central, ddof=1))
    
        
    #noise_mask_conebeam = np.broadcast_to(noise_mask_test[None, :, :], cone_geometry["volume_shape"]).copy()
    #mask_container_conebeam = np.broadcast_to(mask_container_test[None, :, :], cone_geometry["volume_shape"]).copy()

    
    t0 = time.time()  # start segmentation timer
    material_masks_test, selected_thresholds = ljr_segment(  # estimate material masks B_j and their selected intensity thresholds
        log_projection_test_central,
        detector_pixel_size,
        gamma_jei*gam_sc1,
        theta_degrees_test,
        noise_mask_test,
        mask_container = mask_container_test,
        threshold_step=threshold_step,
        filter_fraction_by_axis=filter_fraction_by_axis,
        n_std=n_std,
        moving_window_size=moving_window_size,
        window_density_threshold=window_density_threshold
        
    )
    print("ljr_segment time =", time.time() - t0)
    

    
    mask_container_test_3D = np.broadcast_to(mask_container_test[None, :, :], (log_projection_test.shape[0], *mask_container_test.shape)).copy()
    t0 = time.time()
    material_masks_3D, selected_thresholds_3D = ljr_segment_3D(  # 3D stack version, masks have shape (Ns, Nx, Nx)
        log_projection_test,
        detector_pixel_size,
        gamma_jei*gam_sc2,
        theta_degrees_test,
        noise_mask_test,
        mask_container = mask_container_test_3D,
        threshold_step=threshold_step,
        filter_fraction_by_axis=filter_fraction_by_axis,
        n_std=n_std2,
        moving_kernel_size_3d=moving_kernel_size_3d,
        window_density_threshold=window_density_threshold
    )
    print("ljr_segment_3D time =", time.time() - t0) 
    

    if 0:
        t0 = time.time()
        material_masks_conebeam, selected_thresholds_conebeam = ljr_segment_conebeam(  # 3D stack version, masks have shape (Ns, Nx, Nx)
            log_projection_test,
            detector_pixel_size,
            gamma_jei,
            theta_degrees_test,
            noise_mask_conebeam,
            cone_geometry,
            mask_container=mask_container_conebeam,
            threshold_step=threshold_step,
            filter_fraction_by_axis=filter_fraction_by_axis,
            n_std=n_std,
            moving_window_size=moving_window_size,
            window_density_threshold=window_density_threshold
        )
        print("ljr_segment_conebeam_3D time =", time.time() - t0)
    



    material_masks_3D_central = [current_mask[central_slice_index] for current_mask in material_masks_3D]

    # print("num masks old/3D/conebeam =", len(material_masks_test), len(material_masks_3D), len(material_masks_conebeam))
    print("num masks old/3D =", len(material_masks_test), len(material_masks_3D))
    # print("thresholds old =", np.asarray(selected_thresholds))
    # print("thresholds 3D =", np.asarray(selected_thresholds_3D))
   #  print("thresholds conebeam =", np.asarray(selected_thresholds_conebeam))

    masks_test = np.stack(material_masks_test).astype(float)

    if len(material_masks_test) == len(material_masks_3D):
        masks_3D_central = np.stack([current_mask[central_slice_index] for current_mask in material_masks_3D]).astype(float)
        print("central slice mask err_n old/3D =", err_n(masks_test, masks_3D_central))
        if len(selected_thresholds) == len(selected_thresholds_3D): print("threshold err_n old/3D =", err_n(np.asarray(selected_thresholds), np.asarray(selected_thresholds_3D)))
    else:
        print("Different number of masks old/3D, err_n skipped.")

    #if len(material_masks_test) == len(material_masks_conebeam):
    #    masks_conebeam_central = np.stack([current_mask[central_slice_index] for current_mask in material_masks_conebeam]).astype(float)
     #   print("central slice mask err_n old/conebeam =", err_n(masks_test, masks_conebeam_central))
     #   if len(selected_thresholds) == len(selected_thresholds_conebeam): print("threshold err_n old/conebeam =", err_n(np.asarray(selected_thresholds), np.asarray(selected_thresholds_conebeam)))
   # else:
      #  print("Different number of masks old/conebeam, err_n skipped.")    

    t0 = time.time()  # start thickness timer
    thickness_sinograms, thickness_derivatives = ljr_thickness(material_masks_test, detector_pixel_size, theta_degrees_test)  # compute T_j and dT_j/dx from B_j
    print("ljr_thickness time =", time.time() - t0)  # report thickness computation time

    thickness_sinograms_3D, thickness_derivatives_3D = ljr_thickness(material_masks_3D_central, detector_pixel_size, theta_degrees_test)  # compute T_j and dT_j/dx from B_j
    
    t0 = time.time()  # start thickness timer
    thickness_sinograms_3D_2, thickness_derivatives_3D_2 = ljr_thickness_3D(material_masks_3D, detector_pixel_size, theta_degrees_test)
    print("ljr_thickness time =", time.time() - t0)  # report thickness computation time
    
    central_slice_index = int(np.ceil(log_projection_test.shape[0] / 2.0) - 1)

    # fit_weights_3D = np.sum(thickness_sinograms_3D_2, axis=0)  # (Ntheta, Nx, Ns)
    # fit_weights_3D = 1.0 + 4.0 * fit_weights_3D / np.max(fit_weights_3D)
    #fit_weights = fit_weights_3D[..., central_slice_index]  # (Ntheta, Nx)
 
    t0 = time.time()  # start LJR coefficient-estimation timer
    X_est_r = estimate_material_coefficients(-log_projection_test_unscaled[central_slice_index], thickness_sinograms, thickness_derivatives, j_ei, k_mean2)  # solve for beta_j and delta_j
    print("estimate_material_coefficients time =", time.time() - t0)  # report linear solve time

    X_est_r_3D = estimate_material_coefficients(-log_projection_test_unscaled[central_slice_index], thickness_sinograms_3D, thickness_derivatives_3D, j_ei, k_mean2)  # solve for beta_j and delta_j
    
    X_est_r_3D_2 = estimate_material_coefficients_3D(-log_projection_test_unscaled, thickness_sinograms_3D_2, thickness_derivatives_3D_2, j_ei, k_mean2)
    
    
    Nv = X_est_r.shape[1] // 2  # number of material masks, assuming beta and delta coefficients are both present

    t0 = time.time()  # start projection-composition timer
    PROJ_ref_int_r, PROJ_abs_r = compose_ljr_projection_components(X_est_r, thickness_sinograms)  # build delta-weighted and beta-weighted projection components
    print("compose_ljr_projection_components time =", time.time() - t0)  # report projection-composition time

    PROJ_ref_int_r_3D, PROJ_abs_r_3D = compose_ljr_projection_components(X_est_r_3D, thickness_sinograms_3D)  # build delta-weighted and 
    PROJ_ref_int_r_3D_2, PROJ_abs_r_3D_2 = compose_ljr_projection_components_3D(X_est_r_3D_2, thickness_sinograms_3D_2)  # build delta-
    
    t0 = time.time()  # start final reconstruction timer
    s_ref_est = astra_iradon_parallel(PROJ_ref_int_r, theta_degrees_test, PROJ_ref_int_r.shape[-1]) / detector_pixel_size  # reconstruct delta-like map from refraction integral projection
    s_abs_est = astra_iradon_parallel(PROJ_abs_r, theta_degrees_test, PROJ_abs_r.shape[-1]) / detector_pixel_size  # reconstruct beta-like map from absorption projection

    s_ref_est_3D = astra_iradon_parallel(PROJ_ref_int_r_3D, theta_degrees_test, PROJ_ref_int_r_3D.shape[-1]) / detector_pixel_size  # reconstruct delta-like map from refraction integral projection
    s_abs_est_3D = astra_iradon_parallel(PROJ_abs_r_3D, theta_degrees_test, PROJ_abs_r_3D.shape[-1]) / detector_pixel_size  # reconstruct 

    s_ref_est_3D_2 = astra_iradon_parallel_3D(PROJ_ref_int_r_3D_2, theta_degrees_test) / detector_pixel_size
    s_abs_est_3D_2 = astra_iradon_parallel_3D(PROJ_abs_r_3D_2, theta_degrees_test) / detector_pixel_size

    s_ref_est_3D_2_central = s_ref_est_3D_2 if s_ref_est_3D_2.ndim == 2 else s_ref_est_3D_2[..., central_slice_index]
    s_abs_est_3D_2_central = s_abs_est_3D_2 if s_abs_est_3D_2.ndim == 2 else s_abs_est_3D_2[..., central_slice_index]

    '''
    print("s_ref_est_3D err_n =", err_n(s_ref_est, s_ref_est_3D))
    print("s_abs_est_3D err_n =", err_n(s_abs_est, s_abs_est_3D))
    print("s_ref_est_3D_2 err_n =", err_n(s_ref_est_3D, s_ref_est_3D_2_central))
    print("s_abs_est_3D_2 err_n =", err_n(s_abs_est_3D, s_abs_est_3D_2_central))

    print("s_ref_est_3D max abs diff =", np.max(np.abs(s_ref_est - s_ref_est_3D)))
    print("s_abs_est_3D max abs diff =", np.max(np.abs(s_abs_est - s_abs_est_3D)))
    print("s_ref_est_3D_2 max abs diff =", np.max(np.abs(s_ref_est_3D - s_ref_est_3D_2_central)))
    print("s_abs_est_3D_2 max abs diff =", np.max(np.abs(s_abs_est_3D - s_abs_est_3D_2_central)))
    ''' 
    
     ## Plotting and saving files
    ref_png = sample_folder / "s_ref_est_test.png"; abs_png = sample_folder / "s_abs_est_test.png"; #mat_file = sample_folder / "final_reconstructions_test.mat"  # output paths for figures and MATLAB file
    
    ref_png_3D = sample_folder / "s_ref_est_3D.png"; abs_png_3D = sample_folder / "s_abs_est_3D.png"; 
    
    sl = int(log_projection_test.shape[0]); ii = slice_indices[0]; mw = int(moving_window_size); kz = int(moving_kernel_size_3d[2]); wd = int(round(window_density_threshold * 100)); sd = int(round(n_std2 * 100)); ts = int(round(threshold_step * 1e12)); ff = int(round(filter_fraction_by_axis[-1] * 100)); ag = int(ang_step)
    mat_file_3D = sample_folder / f"P_recs_sl_{sl}_ii_{ii}_mw_{mw}_kz_{kz}_wd_{wd}_sd_{sd}_ts_{ts}_ff_{ff}_ag_{ag}.mat"

    plt.figure(1); plt.clf(); plt.imshow(s_ref_est, cmap="gray", vmin=3.3e-7, vmax=4.85e-7); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(ref_png, dpi=200, bbox_inches="tight")  # save delta-like reconstruction image
    plt.figure(2); plt.clf(); plt.imshow(s_abs_est, cmap="gray", vmin=1.6e-10, vmax=4.5e-10); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(abs_png, dpi=200, bbox_inches="tight")  # save beta-like reconstruction image
    plt.figure(3); plt.clf(); plt.imshow(s_ref_est_3D, cmap="gray", vmin=3.3e-7, vmax=4.85e-7); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(ref_png_3D, dpi=200, bbox_inches="tight")  # save 3D-segmentation delta-like reconstruction image
    plt.figure(4); plt.clf(); plt.imshow(s_abs_est_3D, cmap="gray", vmin=1.6e-10, vmax=4.5e-10); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(abs_png_3D, dpi=200, bbox_inches="tight")  # save 3D-segmentation beta-like reconstruction image

    #print("saved ref:", ref_png); print("saved abs:", abs_png)
    #print("saved ref 3D:", ref_png_3D); print("saved abs 3D:", abs_png_3D)

    #savemat(mat_file, {"s_ref_est": s_ref_est, "s_abs_est": s_abs_est})  # save final 2D-segmentation reconstructions for MATLAB analysis
    savemat(mat_file_3D, {"s_ref_est": s_ref_est, "s_ref_est_3D": s_ref_est_3D, "s_ref_est_3D_2": s_ref_est_3D_2, "s_abs_est": s_abs_est, "s_abs_est_3D": s_abs_est_3D,"s_abs_est_3D_2": s_abs_est_3D_2, "ang_step":ang_step})  # save final 3D-segmentation reconstructions for MATLAB analysis
    #print("saved mat:", mat_file); 
    print("saved mat file:", mat_file_3D)

    if 0:
        plt.figure(1); plt.clf(); plt.imshow(s_ref_est, cmap="gray", vmin=3.3e-7, vmax=4.85e-7); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(ref_png, dpi=200, bbox_inches="tight")  # save delta-like reconstruction image
        plt.figure(2); plt.clf(); plt.imshow(s_abs_est, cmap="gray", vmin=1.6e-10, vmax=4.5e-10); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(abs_png, dpi=200, bbox_inches="tight")  # save beta-like reconstruction image

        # print("saved ref:", ref_png) 
        # print("saved abs:", abs_png)  

       # savemat(mat_file, {"s_ref_est": s_ref_est, "s_abs_est": s_abs_est})  # save final reconstructions for MATLAB analysis
        #print("saved mat:", mat_file_3D)  # report saved .mat path
        #print("file name: ",)


if __name__ == "__main__":  
    main()  # launch the pipeline
    
    
    
    
    
    
    
    
    