import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from ljr_xpci.segmentation import ljr_segment
from ljr_xpci.radon_tools import ljr_thickness
from ljr_xpci.ljr_solver import estimate_material_coefficients
from ljr_xpci.plotting import show_image
from ljr_xpci.ljr_solver import estimate_material_coefficients, compose_ljr_projection_components
from ljr_xpci.radon_tools import _astra_iradon_parallel
from ljr_xpci.radon_tools_cached import astra_iradon_parallel_cached
from ljr_xpci.utility_functions import err_n
from scipy.io import savemat
from ljr_xpci.radon_tools_cached import ljr_thickness_cached, astra_iradon_parallel_cached

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
sys.path.insert(0, str(CONFIG_DIR))

from local_paths import get_default_data_root

def main():
    project_folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_folder / "configs"))




    parser = argparse.ArgumentParser(description="Run the LJR pipeline on the Khush dataset.")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to the folder containing the project data.")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root is not None else get_default_data_root()
    if data_root is None: raise RuntimeError("Data root was not provided and this computer was not recognized. Use --data-root /path/to/data.")

    sample_folder = data_root / "pd" / "Khush_folder"
    data_file = sample_folder / "DATA.tif"
    results_file = sample_folder / "khush_results2.npz"




    data = tiff.imread(data_file).astype(float)
    data[np.isinf(data)] = 0

    #print("data_file =", data_file)
    #print("data.shape =", data.shape)


    n_slices_total, number_x_pixels, number_angles = data.shape

    k_mean2 = 8.8e10
    detector_pixel_size = 6e-6
    theta_degrees = np.linspace(0.0, 360.0, number_angles + 1)[:-1]
    gamma_jei = -21111744.110502686 * 0.2e-11
    j_ei = -21000.0

    #slice_indices = np.arange(4, 7)  # Python indices for MATLAB 5:7
    slice_indices = np.arange(4, 5)  # Python indices for MATLAB 5:7

    N_slice = slice_indices.size

    log_projection = data[slice_indices, :, :]
    log_projection = np.transpose(log_projection, (0, 2, 1))  # shape (Ns, Ntheta, Nx)
    log_projection[np.isinf(log_projection)] = 0

    #print("log_projection.shape =", log_projection.shape)
    ##print("theta_degrees.shape =", theta_degrees.shape)

    #print("Python mean/min/max =", np.mean(log_projection), np.min(log_projection), np.max(log_projection))


    n_detectors = log_projection.shape[-1]

    pix_x = np.arange(1, n_detectors + 1, dtype=float)[:, None]
    pix_y = np.arange(1, n_detectors + 1, dtype=float)[None, :]

    c_x, c_y = 1300.0, 1200.0
    noise_mask = (pix_x - c_x)**2 + (pix_y - c_y)**2 > 1100.0**2

    c_x2, c_y2 = 1338.0, 1286.5
    mask_container_ext = (pix_x - c_x2)**2 + (pix_y - c_y2)**2 < 887.0**2

    c_x3, c_y3 = 1331.0, 1268.0
    mask_container_int = (pix_x - c_x3)**2 + (pix_y - c_y3)**2 < 745.0**2

    #print("noise_mask.shape =", noise_mask.shape)
    #print("noise_mask mean/min/max =", np.mean(noise_mask), np.min(noise_mask.astype(int)), np.max(noise_mask.astype(int)))
    #print("noise_mask nonzero =", np.count_nonzero(noise_mask))


    from ljr_xpci.preprocessing import filter_ft

    import time

    ang_step = 1
    log_projection_test = log_projection[:, ::ang_step, ::ang_step] / (2.0 * k_mean2)  # shape (Ns, Ntheta_test, Nx_test)
    log_projection_test_unscaled = log_projection[:, ::ang_step, ::ang_step]  # shape (Ns, Ntheta_test, Nx_test)
    theta_degrees_test = theta_degrees[::ang_step]
    noise_mask_test = noise_mask[::ang_step, ::ang_step]


    t0 = time.time()
    material_masks_test,selected_thresholds = ljr_segment(
        log_projection_test,
        detector_pixel_size,
        gamma_jei,
        noise_mask_test,
        theta_degrees_test,
        threshold_step= 1e-10,
        filter_fraction_by_axis=(None,None, 0.4),
        n_std=3,
    )

    print("ljr_segment time =", time.time() - t0)
    print("num masks =", len(material_masks_test))

    plot_results = False
    if plot_results:
        for iv, m in enumerate(material_masks_test):
            print(f"mask {iv+1} size = {m.shape}, beta = {selected_thresholds[iv]}, mean/min/max/nnz =", np.mean(m), np.min(m.astype(int)), np.max(m.astype(int)), np.count_nonzero(m))


    # t0 = time.time()
    # thickness_sinograms, thickness_derivatives = ljr_thickness(material_masks_test, detector_pixel_size, theta_degrees_test)
    # print("ljr_thickness time =", time.time() - t0)

    t0 = time.time()
    thickness_sinograms, thickness_derivatives = ljr_thickness_cached(material_masks_test, detector_pixel_size, theta_degrees_test)
    print("ljr_thickness_cached time =", time.time() - t0)

    # thickness_error = err_n(thickness_sinograms, thickness_sinograms_cached)
    # thickness_derivative_error = err_n(thickness_derivatives, thickness_derivatives_cached)

    # print("max abs thickness diff =", np.max(np.abs(thickness_sinograms - thickness_sinograms_cached)))
    # print("max abs derivative diff =", np.max(np.abs(thickness_derivatives - thickness_derivatives_cached)))
    # print(f"relative error thickness = {thickness_error:.15g}")
    # print(f"relative error thickness derivative = {thickness_derivative_error:.15g}")

    # diff_D = np.abs(thickness_derivatives - thickness_derivatives_cached)
    # idx = np.unravel_index(np.argmax(diff_D), diff_D.shape)

    # print("max derivative diff index =", idx)
    # print("max derivative diff value =", diff_D[idx])
    # print("left boundary diff =", np.max(diff_D[:, :, 0]))
    # print("interior diff =", np.max(diff_D[:, :, 1:-1]))
    # print("right boundary diff =", np.max(diff_D[:, :, -1]))

    if plot_results:
        for iv in range(len(thickness_sinograms)):
            m1 = thickness_sinograms[iv, :, :]
            m2 = thickness_derivatives[iv, :, :]
            print(f"Thickness {iv+1}: {np.mean(m1):.15g} {np.min(m1):.15g} {np.max(m1):.15g} {np.std(m1, ddof=1):.15g}, DThick {np.mean(m2):.15g} {np.min(m2):.15g} {np.max(m2):.15g} {np.std(m2, ddof=1):.15g}")

    is_c = np.ceil(N_slice/2).astype(int)-1  # Python index for MATLAB is_c = 2
    test_mat = False #True# 

    if test_mat:
        from scipy.io import loadmat
        test_data = loadmat("/home/amazzolani/Matlab_drive_scripts/remotedata_Andrea_clio/pd/Khush_folder/LJR_thickness_test.mat")

        T0_r_mat = test_data["T0_r"].astype(float)      # MATLAB shape expected: (1, Nx, Ntheta, Nv)
        DT0_r_mat = test_data["DT0_r"].astype(float)    # MATLAB shape expected: (1, Nx, Ntheta, Nv)

        thickness_sinograms = np.transpose(np.squeeze(T0_r_mat, axis=0), (2, 1, 0))       # Python shape: (Nv, Ntheta, Nx)
        thickness_derivatives = np.transpose(np.squeeze(DT0_r_mat, axis=0), (2, 1, 0))    # Python shape: (Nv, Ntheta, Nx)


    t0 = time.time()
    X_est_r = estimate_material_coefficients(-log_projection_test_unscaled[is_c], thickness_sinograms, thickness_derivatives, j_ei, k_mean2)
    print("estimate_material_coefficients time =", time.time() - t0)

    Nv = X_est_r.shape[1] // 2

    if plot_results:
        for iv in range(Nv):
            m1 = X_est_r[:, iv]
            m2 = X_est_r[:, iv + Nv]
            print(f"Material {iv}: beta {iv+1:.15g}: {np.mean(m1):.15g}, delta {np.mean(m2):.15g}")

    t0 = time.time()
    PROJ_ref_int_r, PROJ_abs_r = compose_ljr_projection_components(X_est_r, thickness_sinograms)
    print("compose_ljr_projection_components time =", time.time() - t0)

    if plot_results:
        print("python: PROJ_ref_int_r.shape =", PROJ_ref_int_r.shape)

        print(f"python: PROJ_ref_int_r = {np.mean(PROJ_ref_int_r)}, {np.min(PROJ_ref_int_r)} ,{np.max(PROJ_ref_int_r)}")
        print(f"python: PROJ_abs_r = {np.mean(PROJ_abs_r)}, {np.min(PROJ_abs_r)} ,{np.max(PROJ_abs_r)}")


    t0 = time.time()
    s_ref_est = astra_iradon_parallel_cached(PROJ_ref_int_r, theta_degrees_test, PROJ_ref_int_r.shape[-1]) / detector_pixel_size
    s_abs_est = astra_iradon_parallel_cached(PROJ_abs_r, theta_degrees_test, PROJ_abs_r.shape[-1]) / detector_pixel_size
    print("Astra reconstructions time =", time.time() - t0)

    if plot_results:
        print("python: s_ref_est.shape =", s_ref_est.shape)
        print("python: s_abs_est.shape =", s_abs_est.shape)

        print(f"s_ref_est mean/min/max/std = {np.mean(s_ref_est):.15g} {np.min(s_ref_est):.15g} {np.max(s_ref_est):.15g} {np.std(s_ref_est, ddof=1):.15g}")
        print(f"s_abs_est mean/min/max/std = {np.mean(s_abs_est):.15g} {np.min(s_abs_est):.15g} {np.max(s_abs_est):.15g} {np.std(s_abs_est, ddof=1):.15g}")    


    ref_png = sample_folder / "s_ref_est2.png"; abs_png = sample_folder / "s_abs_est2.png"; mat_file = sample_folder / "final_reconstructions2.mat"

    plt.figure(1); plt.clf(); plt.imshow(s_ref_est, cmap="gray", vmin=3.3e-7, vmax=4.85e-7); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(ref_png, dpi=200, bbox_inches="tight")
    plt.figure(2); plt.clf(); plt.imshow(s_abs_est, cmap="gray", vmin=1.6e-10, vmax=4.5e-10); plt.axis("equal"); plt.colorbar(); plt.tight_layout(); plt.savefig(abs_png, dpi=200, bbox_inches="tight")

    print("saved ref:", ref_png)
    print("saved abs:", abs_png)    
    
    savemat(mat_file, {"s_ref_est": s_ref_est, "s_abs_est": s_abs_est})
    print("saved mat:", mat_file)


if __name__ == "__main__":
    main()






