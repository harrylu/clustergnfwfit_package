from astropy.io import fits
import os

if __name__ == "__main__":
    hdul = fits.open(os.path.join(os.getcwd(), "src", "clustergnfwfit", "macsj0025.4_Bolocam_surface_brightness_20211005.fits"))
    print(hdul[0].data.shape)