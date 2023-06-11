import beam_utils
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# BEAM_MAP_WIDTH = 17
# FPATH_BEAM_90 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"
# FPATH_BEAM_150 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"

# beam_handler_90 = beam_utils.BeamHandlerACTPol(FPATH_BEAM_90, BEAM_MAP_WIDTH)
# beam_handler_150 = beam_utils.BeamHandlerACTPol(FPATH_BEAM_150, BEAM_MAP_WIDTH)

# # show radial profile and map
# tck_90 = beam_handler_90.beam_spline_tck
# tck_150 = beam_handler_150.beam_spline_tck

# start = 0
# stop = 200
# fig_90, (ax_90_profile, ax_90_map) = plt.subplots(2, 1)
# fig_90.suptitle("90")
# ax_90_profile.plot(np.linspace(start, stop), [scipy.interpolate.splev(r, tck_90) for r in np.linspace(start, stop)])
# ax_90_map.imshow(beam_handler_90.beam_map)

# fig_150, (ax_150_profile, ax_150_map) = plt.subplots(2, 1)
# fig_150.suptitle("150")
# ax_150_profile.plot(np.linspace(start, stop), [scipy.interpolate.splev(r, tck_150) for r in np.linspace(start, stop)])
# ax_150_map.imshow(beam_handler_150.beam_map)

# get CMB beam

MAP_FITS_DIR = "/home/harry/ClusterGnfwFit/map_fits_files"
FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

fits_cmb_path = os.path.join(MAP_FITS_DIR, FNAME_CMB)
hdul = fits.open(fits_cmb_path)
beam_hdu = hdul[2]
Bl = list(beam_hdu.columns['INT_BEAM'].array)
cmb_beam_handler = beam_utils.BeamHandlerPlanckCMB(Bl, 121)
tck_cmb = cmb_beam_handler.beam_spline_tck
cmb_beam_map = cmb_beam_handler.beam_map

start = 0
stop = 1800
fig_cmb, (ax_cmb_profile, ax_cmb_map) = plt.subplots(2, 1)
fig_cmb.suptitle("cmb")
ax_cmb_profile.plot(np.linspace(start, stop), [scipy.interpolate.splev(r, tck_cmb) for r in np.linspace(start, stop)])
ax_cmb_map.imshow(cmb_beam_handler.beam_map)

plt.show()
