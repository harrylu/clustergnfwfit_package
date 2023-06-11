from astropy.io import fits
from astropy.table import Table
import os
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

MAP_FITS_DIR = "/home/harry/ClusterGnfwFit/map_fits_files"
FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb
fpath = os.path.join(MAP_FITS_DIR, FNAME_CMB)
#print(fits.info(fpath))

#hdul = fits.open(fpath)
#print(hdul[1].columns['I_STOKES_INP'].array)
#print(hdul[1].columns['I_STOKES_INP'].coord_inc)

# I_STOKES_INP is column (field) 5
m, header = hp.fitsfunc.read_map(fpath, 5, hdu=1, memmap=True, h=True)
header_dict = {k:v for k,v in header}
print(m.shape)
NSIDE = header_dict['NSIDE']
print(f'NSIDE: {NSIDE}')

# hp.mollview(
#     m,
#     coord=["G", "E"],
#     title="Histogram equalized Ecliptic",
#     unit="mK",
#     norm="hist",
#     min=-1,
#     max=1,
# )
# hp.graticule()
# plt.show()

dec = [-12, -22, -45]  # in degrees, minutes, seconds
ra = [0, 25, 29.9]     # in hours, minutes, seconds

def hms_to_deg(hours, minutes, seconds):
    return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
def dms_to_deg(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / (60 ** 2)

deg_ra = hms_to_deg(*ra)
deg_dec = dms_to_deg(*dec)

vec = hp.pixelfunc.ang2vec(deg_dec, deg_ra, lonlat=True)
ipix = hp.query_disc(NSIDE, vec, np.deg2rad(1))
plt.imshow(ipix)
plt.show()