# generate r_y degenerate maps

import ellipsoid_model
import numpy as np
import beam_utils
import matplotlib.pyplot as plt
from extract_maps import extract_maps
import os
import plot_utils
from matplotlib import cm

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

  # these fields will vary depending on the cluster
  dec = [-12, -22, -45]  # in degrees, minutes, seconds
  ra = [0, 25, 29.9]     # in hours, minutes, seconds
  map_radius = 12.5  # arcminutes
  sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_map_err, beam_handler_bolocam = extract_maps(fpath_dict,
              dec, ra, map_radius)
  print(f"ACT map size: {sfl_90.shape}")

  R500 = 200
  theta, P0_150, P0_90, r_x, r_y, r_z, x_offset, y_offset, c_90, c_150, P0_bolocam, c_bolocam = [ 26.47100346,  -2.70946983,  -4.18780089, 386.09227685, 563.29419359,
 466.35130292, -33.30293239,  -9.24344092,  29.83170751,  18.78258829,
  -2.36758264,  20.15412503]
  r_z = np.sqrt(r_x * r_y)
  img_height = 20
  img_width = 20

  gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

  psf_padding_act = beam_handler_150.get_pad_pixels()
  # can use this to make the 90 model beause only P0 is different
  model_act_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, x_offset, y_offset,
                      (sfl_90.shape[0] + psf_padding_act)*3, (sfl_90.shape[1] + psf_padding_act)*3)
  # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
  model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

  model_150_no_c = model_act_no_c * P0_150
  model_90_no_c = model_act_no_c * P0_90

  model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
  model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)

  psf_padding_bolocam = beam_handler_bolocam.get_pad_pixels()
  # eval bolocam at 5 arcsecond res, rebin to 20
  model_bolo_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 5, x_offset, y_offset,
                                                                  (bolocam_map.shape[0] + psf_padding_bolocam)*4, (bolocam_map.shape[1] + psf_padding_bolocam)*4)
  model_bolo_no_c = ellipsoid_model.rebin_2d(model_bolo_no_c, (4, 4))
  # use 150 to make bolocam model because only P0 is different
  model_bolo_no_c = model_bolo_no_c * P0_bolocam

  model_bolocam = beam_handler_bolocam.convolve2d(model_bolo_no_c + c_bolocam, cut_padding=True)
  

  plt.figure("90 model")
  plot_utils.imshow_gaussian_blur_default(1.5, 1.5, model_90, -100, 100)

  plt.figure("150 model")
  plot_utils.imshow_gaussian_blur_default(1.5, 1.5, model_150, -100, 100)

  plt.figure("bolocam model")
  plot_utils.imshow_gaussian_blur_default(1.5, 1.5, model_bolocam, -100, 100)


  
  plt.figure("sfl 90")
  plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_90, -100, 100)

  plt.figure("sfl 150")
  plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_150, -100, 100)

  plt.figure("bolocam")
  plot_utils.imshow_gaussian_blur_default(1.5, 1.5, bolocam_map, -100, 100)


  plt.figure("sfl 90 noise")
  plt.imshow(err_90, cmap=cm.coolwarm, vmin=-100, vmax=100)

  plt.figure("sfl 150 noise")
  plt.imshow(err_150, cmap=cm.coolwarm, vmin=-100, vmax=100)

  plt.figure("bolocam noise")
  plt.imshow(bolocam_map_err, cmap=cm.coolwarm, vmin=-100, vmax=100)
  

  plt.show()