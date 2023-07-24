import numpy as np
import os

import ellipsoid_model_old
import ellipsoid_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    MAP_FITS_DIR = "/home/harry/clustergnfwfit_package/data/map_fits_files"
    FNAME_BRIGHTNESS_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_map_srcfree.fits'
    FNAME_NOISE_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_ivar.fits'
    FNAME_BRIGHTNESS_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits'
    FNAME_NOISE_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_ivar.fits'
    FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

    BOLOCAM_DIR = '/home/harry/clustergnfwfit_package/data/MACS_J0025.4-1222'
    FNAME_FILTERED = 'filtered_image.fits'
    FNAME_RMS = 'filtered_image_rms.fits'
    FNAME_TRANSFER = 'filtered_image_signal_transfer_function.fits'

    FPATH_BEAM_150 = r"/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"
    FPATH_BEAM_90 = r"/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"

    # CLUSTER_NAME = 'MACSJ0025.4'

    # file paths: these fields will stay the same regardless of cluster
    fpath_dict = {
        'brightness_150': os.path.join(MAP_FITS_DIR, FNAME_BRIGHTNESS_150),
        'noise_150': os.path.join(MAP_FITS_DIR, FNAME_NOISE_150),
        'brightness_90': os.path.join(MAP_FITS_DIR, FNAME_BRIGHTNESS_90),
        'noise_90': os.path.join(MAP_FITS_DIR, FNAME_NOISE_90),
        'cmb': os.path.join(MAP_FITS_DIR, FNAME_CMB),
        'beam_150': FPATH_BEAM_150,
        'beam_90': FPATH_BEAM_90,

        'bolocam_filtered': os.path.join(BOLOCAM_DIR, FNAME_FILTERED),
        'bolocam_noise': os.path.join(BOLOCAM_DIR, FNAME_RMS),
        'bolocam_transfer': os.path.join(BOLOCAM_DIR, FNAME_TRANSFER),
    }


    R500 = 200
    theta, P0_150, P0_90, r_x, r_y, r_z, x_offset, y_offset, c_90, c_150, P0_bolocam, c_bolocam = [ 26.47100346,  -2.70946983,  -4.18780089, 386.09227685, 563.29419359,
    466.35130292, -33.30293239,  -9.24344092,  29.83170751,  18.78258829,
    -2.36758264,  20.15412503]
    r_z = np.sqrt(r_x * r_y)
    img_height = 20
    img_width = 20

    import time

    timer = time.time()
    gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

    model_act_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, x_offset, y_offset,
                        (img_height)*3, (img_width)*3)
    # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
    model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))
    print(f"New: {time.time() - timer}")

    timer = time.time()
    gnfw_s_xy_sqr_old = ellipsoid_model_old.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)
    
    model_act_no_c_old = ellipsoid_model_old.eval_pixel_centers_use_interp(gnfw_s_xy_sqr_old, theta, r_x, r_y, 10, x_offset, y_offset,
                        (img_height)*3, (img_width)*3)
    # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
    model_act_no_c_old = ellipsoid_model_old.rebin_2d(model_act_no_c_old, (3, 3))
    print(f"Old: {time.time() - timer}")

    plt.figure('model')
    plt.imshow(model_act_no_c)
    plt.figure('model old')
    plt.imshow(model_act_no_c_old)
    plt.figure('diff')
    plt.imshow(model_act_no_c - model_act_no_c_old)
    plt.show()