import os
from astropy.io import fits
import numpy as np

from pixell import enmap, utils, reproject
import matplotlib.pyplot as plt
from matplotlib import cm
import healpy as hp
import scipy.fft
import scipy.fftpack

import plot_utils
import deconvolution
import beam_utils

from astropy.modeling.functional_models import Gaussian2D

MAP_FITS_DIR = "/home/harry/clustergnfwfit_package/data/map_fits_files"
BOLOCAM_DIR = '/home/harry/clustergnfwfit_package/data/MACS_J0025.4-1222'
FNAME_FILTERED = 'filtered_image.fits'
FNAME_RMS = 'filtered_image_rms.fits'
FNAME_TRANSFER = 'filtered_image_signal_transfer_function.fits'
FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

fpath_dict = {
    'filtered': os.path.join(BOLOCAM_DIR, FNAME_FILTERED),
    'noise': os.path.join(BOLOCAM_DIR, FNAME_RMS),
    'transfer': os.path.join(BOLOCAM_DIR, FNAME_TRANSFER),
    'cmb': os.path.join(MAP_FITS_DIR, FNAME_CMB),
}

# these fields will vary depending on the cluster
dec = [-12, -22, -45]  # in degrees, minutes, seconds
ra = [0, 25, 29.9]     # in hours, minutes, seconds

def hms_to_deg(hours, minutes, seconds):
    return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
def dms_to_deg(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / (60 ** 2)

decimal_dec = dms_to_deg(*dec)
decimal_ra = hms_to_deg(*ra)
coords = [np.deg2rad([decimal_dec, decimal_ra])]

# field 5 is inpaint
# field 0 not inpaint
hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
# get CMB beam
hdul = fits.open(fpath_dict['cmb'])
beam_hdu = hdul[2]
Bl = list(beam_hdu.columns['INT_BEAM'].array)
cmb_radius_deg = 0.503
# 20 arcsecond resolution
# we need to add 10 arcsecond offset, then cut afterwards to match bolocam's 42 x 42 pixels (no center pixel)
ten_arcseconds_deg = 0.00277778
enmap_deconvolved_cmb = deconvolution.get_deconvolved_map_odd(hp_map, Bl, decimal_dec + ten_arcseconds_deg, decimal_ra + ten_arcseconds_deg, cmb_radius_deg, res=1/3, lmax=2000)

# 11 pixels wide
# sum equals 1 for psf
# convolve with Bolocam psf
header = fits.open(fpath_dict['filtered'])[0].header
# beam is approx gaussian, fwhm in degrees
bolocam_beam_fwhm = header['BMAJ']

bolocam_beam_handler = beam_utils.BeamHandlerBolocam(bolocam_beam_fwhm, 11)
plt.figure('bolocam beam map')
plt.imshow(bolocam_beam_handler.beam_map)


sfl_cmb = reproject.thumbnails(enmap_deconvolved_cmb, coords, r=cmb_radius_deg*60*utils.arcmin, res=1/3*utils.arcmin, proj='sfl')[0]
print(sfl_cmb.wcs)
sfl_cmb = bolocam_beam_handler.convolve2d(sfl_cmb)
center_pix = np.array(sfl_cmb.shape) // 2
sfl_cmb = sfl_cmb[center_pix[0]-20:center_pix[0]+22, center_pix[1]-20:center_pix[1]+22]
plt.figure('sfl cmb before filter')
plt.imshow(sfl_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)
plt.figure('sfl cmb magnitude before filter')
plt.imshow(np.abs(scipy.fft.fft2(sfl_cmb)), cmap=cm.coolwarm)


# apply hanning

hanning = np.outer(np.hanning(42), np.hanning(42))
hanning /= np.mean(hanning)
plt.figure("hanning")
plt.imshow(hanning)
sfl_cmb *= hanning


# filter cmb
transfer_function_hdul = fits.open(fpath_dict['transfer'])
signal_transfer_function_fft = transfer_function_hdul[0].data + 1j * transfer_function_hdul[1].data
sfl_cmb = np.real(scipy.fft.ifft2(scipy.fft.fft2(sfl_cmb) * signal_transfer_function_fft))

plt.figure('signal transfer magnitude')
plt.imshow(np.abs(signal_transfer_function_fft), cmap=cm.coolwarm)

plt.figure('sfl cmb after filter')
plt.imshow(sfl_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)

plt.figure('gaussian filtered')
y, x = np.mgrid[:sfl_cmb.shape[0], :sfl_cmb.shape[1]]
gaussian = Gaussian2D.evaluate(x, y, 1, sfl_cmb.shape[1]//2, sfl_cmb.shape[0]//2, 4, 4, 0)
gaussian_filtered = np.real(scipy.fft.ifft2(scipy.fft.fft2(gaussian) * signal_transfer_function_fft))
plt.imshow(gaussian_filtered)
plt.show()


# print(fits.open(fpath_dict['filtered'])[0].header)

# these are in SFL projection
enmap_filtered = enmap.read_fits(fpath_dict['filtered'])
# wcs likely incorrect when read in this way so dont use the wcs

plt.figure('bolocam enmap blurred')
print(enmap_filtered.wcs)
plt.imshow(enmap_filtered, cmap=cm.coolwarm, vmin=-100, vmax=100)
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_filtered, vmin=-100, vmax=100)

plt.figure('enmap - cmb blurred')
plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_filtered - sfl_cmb, vmin=-100, vmax=100)


plt.show()