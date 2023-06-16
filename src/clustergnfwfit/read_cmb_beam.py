import os
from astropy.io import fits
import healpy as hp
from pixell import enmap, reproject, utils
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from extract_maps import extract_maps
import beam_utils
from scipy import fft, fftpack
import scipy.signal
import plot_utils
from skimage import restoration
import deconvolution

plt.rcParams['image.cmap'] = 'coolwarm'

MAP_FITS_DIR = "/home/harry/clustergnfwfit_package/data/map_fits_files"
FNAME_BRIGHTNESS_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_map_srcfree.fits'
FNAME_NOISE_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_ivar.fits'
FNAME_BRIGHTNESS_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits'
FNAME_NOISE_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_ivar.fits'
FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

FPATH_BEAM_150 = r"/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"
FPATH_BEAM_90 = r"/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"


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
map_radius = 12.5  # arcminutes

def hms_to_deg(hours, minutes, seconds):
    return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
def dms_to_deg(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / (60 ** 2)

decimal_dec = dms_to_deg(*dec)
decimal_ra = hms_to_deg(*ra)
coords = [np.deg2rad([decimal_dec, decimal_ra])]

# pixel = 30" = 30/60 ' = 30 / (3600) deg
# pixel = 1 / 120 deg
# pixel_width = 51
# deg_w = pixel_width / 120
# ^ not used

# deg = 60 arcmin, arcmin = 1/60 deg
# so a arcmins = (1/60 deg) / arcmin = a/60 deg
# we add some positive arcmin to the map radius to prevent losing edge data when we reproject (maybe unnecessary?)
# .77 gives odd size of map
# .5 gives even size
# some values like .6 give mismatched axis lengths, which is weird, avoid that
deg_r = (map_radius + .5) / 60

# Create the box and use it to select a submap enmap
box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

# these are in CAR projection
enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
if (enmap_90.shape[0] % 2 == 0 or enmap_90.shape[1] % 2 == 0):
    print("input maps are even")
else:
    print("input maps are odd")
print(f'enmap 90 wcs: {enmap_90.wcs}')

hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
not_deconvolved_cmb_cutout = reproject.enmap_from_healpix(hp_map, enmap_90.shape, enmap_90.wcs, 
                                         ncomp=1, unit=1e-6, rot='gal,equ')[0]
plt.figure('not deconvolved cmb cutout')
plt.imshow(not_deconvolved_cmb_cutout, cmap=cm.coolwarm, vmin=-100, vmax=100)

# I_STOKES_INP is column (field) 5
hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
# reproject from healpix to wcs from fits file (CAR)
# and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen
# extract 1 degree x 1 degree map
# weird, when I was just using enmap_90's wcs, increasing the shape size just adds to the +x,+y of the image
# it looks like enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape goes +x, +y
# so we have to make our own wcs that is 1deg x 1deg and centered at our dec, ra
# res = 1/2 * utils.arcmin
# # want odd shape
# cmb_radius_deg = 0.5
# box = np.deg2rad([[decimal_dec - cmb_radius_deg, decimal_ra - cmb_radius_deg], [decimal_dec + cmb_radius_deg, decimal_ra + cmb_radius_deg]])
# cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=res, proj='car')
# print(f'cmb wcs: {cmb_wcs}')
# enmap_cmb = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
#                                         ncomp=1, unit=1e-6, rot='gal,equ')[0]
# print(f'cmb shape: {cmb_shape}')


# get CMB beam
hdul = fits.open(fpath_dict['cmb'])
beam_hdu = hdul[2]
Bl = list(beam_hdu.columns['INT_BEAM'].array)
cmb_beam_handler = beam_utils.BeamHandlerPlanckCMB(Bl, 121)
cmb_beam_map = cmb_beam_handler.beam_map
cmb_beam_map /= cmb_beam_map[cmb_beam_map.shape[0]//2, cmb_beam_map.shape[1]//2]
plt.figure('original cmb beam')
plt.title('original cmb beam')
plt.imshow(cmb_beam_map, cmap=cm.coolwarm)

# from astropy.modeling.functional_models import Gaussian2D
# from astropy.modeling.fitting import LevMarLSQFitter
# y, x = np.mgrid[:cmb_beam_map.shape[0], :cmb_beam_map.shape[1]]
# fit_p = LevMarLSQFitter()
# model = fit_p(Gaussian2D(x_mean=cmb_beam_map.shape[1]//2, y_mean=cmb_beam_map.shape[0]//2, fixed={'theta': True}), x, y, cmb_beam_map)
# print(model)
# cmb_beam_fitted = model.evaluate(x, y, model.amplitude, model.x_mean, model.y_mean, model.x_stddev, model.y_stddev, model.theta)
# cmb_beam_fitted /= model.amplitude

# plt.figure('original beam - fitted gaussian beam')
# plt.imshow(cmb_beam_map - cmb_beam_fitted)

# #cmb_beam_map = cmb_beam_fitted

# plt.figure('beam fitted gaussian (not used)')
# plt.imshow(cmb_beam_fitted)


beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], 17)
beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], 17)

lmax=2000
if enmap_90.shape[0] % 2 == 0:
    enmap_deconvolved_cmb = deconvolution.get_deconvolved_map_even(hp_map, Bl, decimal_dec, decimal_ra, 0.503, 1/2, lmax)
else:
    enmap_deconvolved_cmb = deconvolution.get_deconvolved_map_odd(hp_map, Bl, decimal_dec, decimal_ra, 0.503, 1/2, lmax)

plt.show()

plt.figure('even cmb')
even_cmb = deconvolution.get_deconvolved_map_even(hp_map, Bl, decimal_dec, decimal_ra, 0.503, 1/2, lmax)
print(f"even shape: {even_cmb.shape}")
plt.imshow(even_cmb, vmin=-100, vmax=100)
plt.figure('odd cmb')
odd_cmb = deconvolution.get_deconvolved_map_odd(hp_map, Bl, decimal_dec, decimal_ra, 0.503, 1/2, lmax)
print(f"even shape: {odd_cmb.shape}")
plt.imshow(odd_cmb, vmin=-100, vmax=100)
plt.show()



print(enmap_deconvolved_cmb.shape)
plt.figure('devonvolved cmb')
plt.imshow(enmap_deconvolved_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)
# plt.figure('deconvolved cmb blurred')
# plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_deconvolved_cmb, -100, 100)
# plt.figure(11)
# plt.title('deconvolved blurred cmb')
# plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_deconvolved_cmb, -100, 100)

# convolve with 90 psf
deconvolved_cmb_90 = beam_handler_90.convolve2d(enmap_deconvolved_cmb)
deconvolved_cmb_150 = beam_handler_150.convolve2d(enmap_deconvolved_cmb)

center_pix = (deconvolved_cmb_90.shape[0] - 1) / 2
crop_amount = (deconvolved_cmb_90.shape[0] - enmap_90.shape[0]) / 2
assert int(crop_amount) == crop_amount, "If this trips, get_deconvolved_map is not correct in even/odd"
crop_amount = int(crop_amount)
deconvolved_cmb_cutout_90 = deconvolved_cmb_90[crop_amount:-crop_amount, crop_amount:-crop_amount]
deconvolved_cmb_cutout_150 = deconvolved_cmb_150[crop_amount:-crop_amount, crop_amount:-crop_amount]
plt.figure('deconvolved_cmb_cutout reconvolved 90')
plt.imshow(deconvolved_cmb_cutout_90, cmap=cm.coolwarm, vmin=-100, vmax=100)
plt.figure('deconvolved_cmb_cutout reconvolved 150')
plt.imshow(deconvolved_cmb_cutout_150, cmap=cm.coolwarm, vmin=-100, vmax=100)


# test deconvolution.py
padding = beam_handler_90.get_pad_pixels()
deconvolved_cmb_90_script = beam_handler_90.convolve2d(deconvolution.get_deconvolved_map(np.array(enmap_90.shape) + padding, hp_map, Bl, decimal_dec, decimal_ra, 0.503, 1/2, lmax))
deconvolved_cmb_150_script = beam_handler_150.convolve2d(deconvolution.get_deconvolved_map(np.array(enmap_150.shape) + padding, hp_map, Bl, decimal_dec, decimal_ra, 0.503, 1/2, lmax))

print(f"enmap_90 shape: {enmap_90.shape}")
plt.figure('diff script - no script 90')
plt.imshow(deconvolved_cmb_90_script - deconvolved_cmb_cutout_90)
plt.figure('diff script - no script 150')
plt.imshow(deconvolved_cmb_150_script - deconvolved_cmb_cutout_150)
print(f"diff - script 90 {np.sum(np.abs(deconvolved_cmb_90_script - deconvolved_cmb_cutout_90))}")
print(f"diff - script 150 {np.sum(np.abs(deconvolved_cmb_150_script - deconvolved_cmb_cutout_150))}")





enmap_90_cmb_subtracted = enmap_90 - deconvolved_cmb_cutout_90
enmap_150_cmb_subtracted = enmap_150 - deconvolved_cmb_cutout_150 

plt.figure('enmap 150 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150, -100, 100)
plt.figure('enmap 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90, vmin=-100, vmax=100)
plt.figure('deconvolved_cmb subtracted 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90_cmb_subtracted, vmin=-100, vmax=100)
plt.figure('deconvolved_cmb subtracted 150 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150_cmb_subtracted, vmin=-100, vmax=100)
print('deconvolved_cmb subtracted')
print('90')
print(f'RMS: {np.sqrt(np.mean(np.square(enmap_90_cmb_subtracted)))}')
print('150')
print(f'RMS: {np.sqrt(np.mean(np.square(enmap_150_cmb_subtracted)))}')

plt.figure('not deconvolved cmb subtracted 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90 - not_deconvolved_cmb_cutout, vmin=-100, vmax=100)
plt.figure('not deconvolved cmb subtracted 150 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150 - not_deconvolved_cmb_cutout, vmin=-100, vmax=100)
print('not deconvolved cmb subtracted')
print('90')
print(f'RMS: {np.sqrt(np.mean(np.square(enmap_90 - not_deconvolved_cmb_cutout)))}')
print('150')
print(f'RMS: {np.sqrt(np.mean(np.square(enmap_150 - not_deconvolved_cmb_cutout)))}')

#plt.figure('deconvolved_cmb subtracted 150 blurred')
#plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150 - deconvolved_cmb_cutout_90, -100, 100)

# dont forget reproject to sfl
radius = map_radius*utils.arcmin
res = 1/2 * utils.arcmin
sfl_90 = reproject.thumbnails(enmap_90_cmb_subtracted, coords, r=radius, res=res, proj='sfl', verbose=True)[0]
sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=res, proj='sfl', verbose=True)[0]
sfl_150 = reproject.thumbnails(enmap_150_cmb_subtracted, coords, r=radius, res=res, proj='sfl', verbose=True)[0]
sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=res, proj='sfl', verbose=True)[0]

plt.figure('sfl 150 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_150, vmin=-100, vmax=100)
plt.figure('sfl 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_90, vmin=-100, vmax=100)
print(sfl_150.wcs)

plt.show()

# double check wcs for sfl; should it be that ra is decreasing in +x and dec is increasing in +y?