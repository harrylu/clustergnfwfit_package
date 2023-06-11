from extract_maps import extract_maps
import os
import plot_utils
from matplotlib import pyplot as plt

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


# these fields will vary depending on the cluster
dec = [-12, -22, -45]  # in degrees, minutes, seconds
ra = [0, 25, 29.9]     # in hours, minutes, seconds
#dec = [0, 0, 0]  # in degrees, minutes, seconds
#ra = [0, 0, 0]     # in hours, minutes, seconds
# ra = [0, 25, 29.9]
map_radius = 60  # arcminutes
R500 = 200  # arcseconds


sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150 = extract_maps(fpath_dict, BEAM_MAP_WIDTH,
                dec, ra, map_radius,
                show_map_plots=False, verbose=False)



plt.figure(0)
plt.title('sfl 150 w/ gaussian blur')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_150, -100, 100)
plt.figure(2)
plt.title('sfl 90 w/ gaussian blur')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_90, -100, 100)
plt.show()