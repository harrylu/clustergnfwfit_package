import numpy as np
from scipy import fftpack
from pixell import enmap, reproject, utils
from astropy.io import fits

import beam_utils
import deconvolution

# res, shape, ra, dec, lmax
# shape must be odd
# decimal_dec, decimal_ra in degrees
# res is float in arcmins
def get_deconvolved_map_fft(hp_map, beam_Bl, decimal_dec, decimal_ra, cmb_radius_deg, res, lmax):
    """Extracts deconvolved CMB map at specifed dec, ra.

    Args:
        hp_map (map): Healpix map from healpy
        beam_Bl (list of float): [Bl(l=0), Bl(l=1),.... Bl(l=lmax)]
        decimal_dec (float): Declination in decimal degrees
        decimal_ra (float): Right ascension in decimal degrees
        cmb_radius_deg (float): CMB map radius in decimal degrees. Must be so that the resulting map is odd.
        res (float): Resolution of map (pixel size) in arcminutes
        lmax (int): lmax of deconvolved map

    Raises:
        Exception: Resulting map must have odd shape

    Returns:
        Pixell ndmap: Deconvolved CMB map with CAR WCS.
    """
    # reproject from healpix to CAR
    # and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen (maybe not anymore?)
    box = np.deg2rad([[decimal_dec - cmb_radius_deg, decimal_ra - cmb_radius_deg], [decimal_dec + cmb_radius_deg, decimal_ra + cmb_radius_deg]])
    # enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape reads +x, +y
    # so we have to make our own wcs that is 1deg x 1deg and centered at our dec, r
    cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')
    enmap_cmb = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
                                            ncomp=1, unit=1e-6, rot='gal,equ')[0]
    # want odd shape for center pixel
    if (cmb_shape[0] % 2 == 0 or cmb_shape[1] % 2 == 0):
        raise Exception(f"Tweak cmb_radius_deg (Trial and error; try values close to {cmb_radius_deg}). Resulting map shape should be odd (for deconvolution) instead of {cmb_shape}.")


    cmb_beam_handler = beam_utils.BeamHandlerPlanckCMB(beam_Bl, cmb_shape[0])
    cmb_beam_map = cmb_beam_handler.beam_map
    # normalize peak of beam
    cmb_beam_map /= cmb_beam_map[cmb_beam_map.shape[0]//2, cmb_beam_map.shape[1]//2]


    fft_enmap_cmb = fftpack.fftshift(enmap.fft(enmap_cmb))
    fft_beam_cmb = fftpack.fftshift(np.abs(enmap.fft(cmb_beam_map)))

    fft_deconvolved_cmb = np.divide(fft_enmap_cmb, fft_beam_cmb)

    # resolution in radians
    rad_res = res * np.pi / (60 * 180) 
    y_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[0], rad_res), (fft_deconvolved_cmb.shape[1], 1)).T
    x_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[1], rad_res), (fft_deconvolved_cmb.shape[0], 1))

    # 2pi * freq to convert from k to l
    l_dist = fftpack.fftshift(np.sqrt(np.square(2*np.pi*y_freq) + np.square(2*np.pi*x_freq)))

    # mask out l > lmax
    fft_deconvolved_cmb[l_dist > lmax] = 0

    enmap_deconvolved_cmb = enmap.ndmap(np.real(enmap.ifft(fftpack.ifftshift(fft_deconvolved_cmb))), cmb_wcs)


    return enmap_deconvolved_cmb