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

plt.rcParams['image.cmap'] = 'coolwarm'

MAP_FITS_DIR = "/home/harry/ClusterGnfwFit/map_fits_files"
FNAME_BRIGHTNESS_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_map_srcfree.fits'
FNAME_NOISE_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_ivar.fits'
FNAME_BRIGHTNESS_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits'
FNAME_NOISE_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_ivar.fits'
FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

FPATH_BEAM_150 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"
FPATH_BEAM_90 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"

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
map_radius = 12.5  # arcminutes
R500 = 200  # arcseconds

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
deg_r = (map_radius + .77) / 60

# Create the box and use it to select a submap enmap
box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

# these are in CAR projection
enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
if (enmap_150.shape[0] % 2 == 0 or enmap_150.shape[1] % 2 == 0):
    raise Exception(f"Tweak map_radius (Trial and error; try values close to {map_radius}). Resulting map shape should be odd (for subtracting deconvolved cmb) instead of {enmap_150.shape}.")

# I_STOKES_INP is column (field) 5
hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
# reproject from healpix to wcs from fits file (CAR)
# and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen
# extract 1 degree x 1 degree map
# weird, when I was just using enmap_90's wcs, increasing the shape size just adds to the +x,+y of the image
# it looks like enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape goes +x, +y
# so we have to make our own wcs that is 1deg x 1deg and centered at our dec, ra
res = 1/2 * utils.arcmin
# want odd shape
cmb_radius_deg = 0.505
box = box = np.deg2rad([[decimal_dec - cmb_radius_deg, decimal_ra - cmb_radius_deg], [decimal_dec + cmb_radius_deg, decimal_ra + cmb_radius_deg]])
cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=res, proj='car')
enmap_cmb = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
                                        ncomp=1, unit=1e-6, rot='gal,equ')[0]
print(f'cmb shape: {cmb_shape}')
if (cmb_shape[0] % 2 == 0 or cmb_shape[1] % 2 == 0):
    raise Exception(f"Tweak cmb_radius_deg (Trial and error; try values close to {cmb_radius_deg}). Resulting map shape should be odd (for deconvolution) instead of {cmb_shape}.")

# testing the weirdness from before:
# # want to cut out center of this
# plt.figure('1deg x 1deg cmb')
# plt.imshow(enmap_cmb, vmin=-100, vmax=100)


# center_pix_y = enmap_cmb.shape[0] // 2
# center_pix_x = enmap_cmb.shape[1] // 2
# cut_amount = enmap_90.shape[0] // 2
# not_deconvolved_cmb_cutout = enmap_cmb[center_pix_y - cut_amount:center_pix_y + cut_amount + 1, center_pix_x - cut_amount:center_pix_x + cut_amount + 1]
# #not_deconvolved_cmb_cutout = enmap_cmb[:enmap_90.shape[0], :enmap_90.shape[1]]
# #print(not_deconvolved_cmb_cutout[not_deconvolved_cmb_cutout.shape[0]//2, not_deconvolved_cmb_cutout.shape[1]//2])
# plt.figure('cutout subtracted no blur')
# plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90 - not_deconvolved_cmb_cutout, vmin=-100, vmax=100)

# plt.figure('cutout')
# plt.imshow(not_deconvolved_cmb_cutout, vmin=-100, vmax=100)

# reproj_cmb = reproject.enmap_from_healpix(hp_map, enmap_90.shape, enmap_90.wcs, 
#                                          ncomp=1, unit=1e-6, rot='gal,equ')[0]

# plt.figure('reprojection subtracted')
# plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90 - reproj_cmb, vmin=-100, vmax=100)

# plt.figure('reproject')
# plt.imshow(reproj_cmb, vmin=-100, vmax=100)

# plt.figure('cutout - reproject / reproject')
# plt.imshow((not_deconvolved_cmb_cutout - reproj_cmb)/reproj_cmb)

# plt.figure('(cutout subtracted) / reprojection subtracted')
# plt.imshow((enmap_90 - not_deconvolved_cmb_cutout) / (enmap_90 - reproj_cmb))
# plt.show()


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

from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
y, x = np.mgrid[:cmb_beam_map.shape[0], :cmb_beam_map.shape[1]]
fit_p = LevMarLSQFitter()
model = fit_p(Gaussian2D(x_mean=cmb_beam_map.shape[1]//2, y_mean=cmb_beam_map.shape[0]//2, fixed={'theta': True}), x, y, cmb_beam_map)
print(model)
cmb_beam_fitted = model.evaluate(x, y, model.amplitude, model.x_mean, model.y_mean, model.x_stddev, model.y_stddev, model.theta)
cmb_beam_fitted /= model.amplitude

plt.figure('original beam - fitted gaussian beam')
plt.imshow(cmb_beam_map - cmb_beam_fitted)

#cmb_beam_map = cmb_beam_fitted

plt.figure('beam fitted gaussian')
plt.imshow(cmb_beam_fitted)

# cmb_beam_map[np.abs(cmb_beam_map) < 0.01] = 0

plt.figure('enmap cmb')
plt.title('enmap cmb')
plt.imshow(enmap_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)

# fft_enmap_cmb = fft.fft2(enmap_cmb)   #scipy.fft.fft2(enmap_cmb)
# fft_beam_cmb = fft.fft2(np.abs(cmb_beam_map))#np.abs(scipy.fft.fft2(cmb_beam_map))

fft_enmap_cmb = fftpack.fftshift(enmap.fft(enmap_cmb))
fft_beam_cmb = fftpack.fftshift(np.abs(enmap.fft(cmb_beam_map))) # enmap.fft(np.abs(cmb_beam_map))
# fft_enmap_cmb *= fft.fft2(beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], fft_beam_cmb.shape[0]).beam_map)

# fft_beam_cmb[fft_beam_cmb < 0.01] = 0.01
fft_deconvolved_cmb = np.divide(fft_enmap_cmb, fft_beam_cmb)
#fft_deconvolved_cmb[fft_beam_cmb < 0.25] = 0

# 30 arcseconds = 0.000145444 radians
y_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[0], 0.000145444), (fft_deconvolved_cmb.shape[1], 1)).T
x_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[1], 0.000145444), (fft_deconvolved_cmb.shape[0], 1))

# 2pi * freq to convert from k to l
l_dist = fftpack.fftshift(np.sqrt(np.square(2*np.pi*y_freq) + np.square(2*np.pi*x_freq)))
plt.figure('l dist')
plt.imshow(l_dist)

# mask out l > 2000
fft_deconvolved_cmb[l_dist > 2000] = 0

# fft_deconvolved_cmb[fft_beam_cmb < 0.1] = 0
# & (fft_beam_cmb < 0.25)

plt.figure("fft_enmap_cmb")
plt.imshow(np.abs(fft_enmap_cmb))
plt.figure("fft_beam_cmb")
plt.title('fft_beam_cmb')
plt.imshow(np.abs(fft_beam_cmb))
plt.figure('fft deconvolved cmb')
plt.imshow(np.abs(fft_deconvolved_cmb))
plt.show()


plt.figure('not deconvolved cmb')
plt.imshow(enmap_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)

# beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], 17)
beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], 17)

# enmap_deconvolved_cmb = np.real(scipy.fft.ifft2(fft_deconvolved_cmb))
# enmap_deconvolved_cmb = np.real(fftpack.ifft2(fft_deconvolved_cmb))
enmap_deconvolved_cmb = np.real(enmap.ifft(fftpack.ifftshift(fft_deconvolved_cmb)))
# enmap_deconvolved_cmb = beam_handler_90.convolve2d(enmap_deconvolved_cmb)
# enmap_deconvolved_cmb = restoration.richardson_lucy(enmap_cmb, cmb_beam_map, clip=False, num_iter=1)
# enmap_deconvolved_cmb, _ = restoration.unsupervised_wiener(enmap_cmb, cmb_beam_map, clip=False)
print(enmap_deconvolved_cmb.shape)
plt.figure('devonvolved cmb')
plt.imshow(enmap_deconvolved_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)
# plt.figure('deconvolved cmb blurred')
# plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_deconvolved_cmb, -100, 100)
# plt.figure(11)
# plt.title('deconvolved blurred cmb')
# plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_deconvolved_cmb, -100, 100)

# convolve with 90 psf
enmap_deconvolved_cmb = beam_handler_90.convolve2d(enmap_deconvolved_cmb)
center_pix_y = enmap_deconvolved_cmb.shape[0] // 2
center_pix_x = enmap_deconvolved_cmb.shape[1] // 2
cut_amount = enmap_90.shape[0] // 2
enmap_deconvolved_cmb_cutout = enmap_deconvolved_cmb[center_pix_y - cut_amount:center_pix_y + cut_amount + 1, center_pix_x - cut_amount:center_pix_x + cut_amount + 1]
plt.figure('deconvolved_cmb_cutout')
plt.imshow(enmap_deconvolved_cmb_cutout, cmap=cm.coolwarm, vmin=-100, vmax=100)
plt.figure('not deconvolved cmb cutout')
center_pix_y = enmap_cmb.shape[0] // 2
center_pix_x = enmap_cmb.shape[1] // 2
not_deconvolved_cmb_cutout = enmap_cmb[center_pix_y - cut_amount:center_pix_y + cut_amount + 1, center_pix_x - cut_amount:center_pix_x + cut_amount + 1]
plt.imshow(not_deconvolved_cmb_cutout, vmin=-100, vmax=100)

#plt.figure('enmap 150 blurred')
#plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150, -100, 100)
plt.figure('enmap 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90, vmin=-100, vmax=100)
plt.figure('deconvolved_cmb subtracted 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90 - enmap_deconvolved_cmb_cutout, vmin=-100, vmax=100)
print('deconvolved_cmb subtracted')
print(f'RMS: {np.sqrt(np.mean(np.square(enmap_90 - enmap_deconvolved_cmb_cutout)))}')
print(f'std: {np.std(enmap_90 - enmap_deconvolved_cmb_cutout)}')

plt.figure('not deconvolved cmb subtracted 90 blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90 - not_deconvolved_cmb_cutout, vmin=-100, vmax=100)
print('not deconvolved cmb subtracted')
print(f'RMS: {np.sqrt(np.mean(np.square(enmap_90 - not_deconvolved_cmb_cutout)))}')
print(f'std: {np.std(enmap_90 - not_deconvolved_cmb_cutout)}')

#plt.figure('deconvolved_cmb subtracted 150 blurred')
#plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150 - enmap_deconvolved_cmb_cutout, -100, 100)


plt.show()
