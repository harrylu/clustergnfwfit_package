# generate r_y degenerate maps

import ellipsoid_model
import numpy as np
import beam_utils
import matplotlib.pyplot as plt
from extract_maps import extract_maps
import os
import plot_utils

if __name__ == "__main__":
    MAP_FITS_DIR = "/home/harry/ClusterGnfwFit/map_fits_files"
    FNAME_BRIGHTNESS_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_map_srcfree.fits'
    FNAME_NOISE_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_ivar.fits'
    FNAME_BRIGHTNESS_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits'
    FNAME_NOISE_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_ivar.fits'
    FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

    # beam of width 17 pixels has smallest values which are within 1% of largest
    BEAM_MAP_WIDTH = 17
    FPATH_BEAM_150 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"
    FPATH_BEAM_90 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"

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
    }

    BEAM_MAP_WIDTH = 17
    FPATH_BEAM_90 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"
    FPATH_BEAM_150 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"

    beam_handler_90 = beam_utils.BeamHandlerACTPol(FPATH_BEAM_90, BEAM_MAP_WIDTH)
    beam_handler_150 = beam_utils.BeamHandlerACTPol(FPATH_BEAM_150, BEAM_MAP_WIDTH)

    R500 = 200
    theta, P0_150, P0_90, r_x, r_y, x_offset, y_offset, c_90, c_150 = [ 1.80729077e+02, -8.14712478e+00, -1.41524045e+01,  5.48282759e+02,
  1.37712615e+02,  4.04895267e-01, -2.65527680e+00,  2.52722829e+01,
  3.84996532e+01]
    r_z = np.sqrt(r_x * r_y)
    img_height = 20
    img_width = 20

    dim_pad = beam_handler_90.get_pad_pixels()
    height = img_height + dim_pad
    width = img_width + dim_pad
    img_90 = ellipsoid_model.eval_pixel_centers(0, P0_90, r_x, r_y, r_z, 10, R500, 0, 0, height*3, width*3)
    img_150 = img_90 * (P0_150/P0_90)
    img_90 += c_90
    img_150 += c_150
    img_90 = ellipsoid_model.rebin_2d(img_90, (3, 3))
    img_150 = ellipsoid_model.rebin_2d(img_150, (3, 3))
    
    img_90 = beam_handler_90.convolve2d(img_90, cut_padding=True)
    plt.figure(1)
    plt.title("90 convolved")
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, img_90, -100, 100)

    img_150 = beam_handler_150.convolve2d(img_150, cut_padding=True)
    plt.figure(2)
    plt.title("150 convolved")
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, img_150, -100, 100)

    # these fields will vary depending on the cluster
    dec = [-12, -22, -45]  # in degrees, minutes, seconds
    ra = [0, 25, 29.9]     # in hours, minutes, seconds
    #dec = [0, 0, 0]  # in degrees, minutes, seconds
    #ra = [0, 0, 0]     # in hours, minutes, seconds
    # ra = [0, 25, 29.9]
    map_radius = 5  # arcminutes
    sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150 = extract_maps(fpath_dict, BEAM_MAP_WIDTH,
                dec, ra, map_radius,
                show_map_plots=False, verbose=False)
    
    plt.figure(90)
    plt.title("sfl 90")
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_90, -100, 100)

    plt.figure(150)
    plt.title("sfl 150")
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_150, -100, 100)

    plt.show()