import numpy as np
from scipy import fftpack
from pixell import enmap, reproject, utils
from astropy.io import fits

import beam_utils
import deconvolution

# specified 
def get_deconvolved_map_odd(hp_map, beam_Bl, decimal_dec, decimal_ra, cmb_radius_deg, res, lmax, proj):
    """Extracts deconvolved CMB map with center pixel at specifed dec, ra. 

    Args:
        hp_map (map): Healpix map from healpy
        beam_Bl (list of float): [Bl(l=0), Bl(l=1),.... Bl(l=lmax)]
        decimal_dec (float): Declination in decimal degrees
        decimal_ra (float): Right ascension in decimal degrees
        cmb_radius_deg (float): CMB map radius in decimal degrees. Must be so that the resulting map is odd.
        res (float): Resolution of map (pixel size) in arcminutes
        lmax (int): lmax of deconvolved map

    Raises:
        Exception: cmb_radius_deg must result in odd shape map (for deconvolution)
        Exception: Resulting map will have odd shape

    Returns:
        Pixell ndmap: Deconvolved CMB map.
    """
    # reproject from healpix to CAR
    # and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen (maybe not anymore?)
    box = np.deg2rad([[decimal_dec - cmb_radius_deg, decimal_ra - cmb_radius_deg], [decimal_dec + cmb_radius_deg, decimal_ra + cmb_radius_deg]])
    # enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape reads +x, +y
    # so we have to make our own wcs that is 1deg x 1deg and centered at our dec, r
    cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')
    print(cmb_shape)
    enmap_cmb = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
                                            ncomp=1, unit=1e-6, rot='gal,equ')[0]
    
    # specified a proj, reproject to that coordinate system
    coords = [np.deg2rad([decimal_dec, decimal_ra])]
    radius = cmb_radius_deg * 60 * utils.arcmin
    print(f'Reprojecting CMB to {proj}')
    enmap_cmb = reproject.thumbnails(enmap_cmb, coords, r=radius, res=res*utils.arcmin, proj=proj)[0]
        

    # want odd shape in order to have center pixel
    if (cmb_shape[0] % 2 == 0 or cmb_shape[1] % 2 == 0):
        raise Exception(f"Tweak cmb_radius_deg (Trial and error; try values close to {cmb_radius_deg}). Resulting map shape should be odd (for deconvolution) instead of {cmb_shape}. Try 0.503 for 1 degree map width.")


    cmb_beam_handler = beam_utils.BeamHandlerPlanckCMB(beam_Bl, enmap_cmb.shape[0])
    cmb_beam_map = cmb_beam_handler.beam_map
    # normalize peak of beam
    cmb_beam_map /= cmb_beam_map[cmb_beam_map.shape[0]//2, cmb_beam_map.shape[1]//2]


    fft_enmap_cmb = fftpack.fftshift(enmap.fft(enmap_cmb))
    fft_beam_cmb = fftpack.fftshift(np.abs(enmap.fft(cmb_beam_map)))

    fft_deconvolved_cmb = np.divide(fft_enmap_cmb, fft_beam_cmb)

    # resolution in radians
    # numpy outer_func would do this more elegantly
    rad_res = res * np.pi / (60 * 180) 
    y_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[0], rad_res), (fft_deconvolved_cmb.shape[1], 1)).T
    x_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[1], rad_res), (fft_deconvolved_cmb.shape[0], 1))

    # 2pi * freq to convert from k to l
    l_dist = fftpack.fftshift(np.sqrt(np.square(2*np.pi*y_freq) + np.square(2*np.pi*x_freq)))

    # mask out l > lmax
    fft_deconvolved_cmb[l_dist > lmax] = 0

    # enmap_deconvolved_cmb = enmap.ndmap(np.real(enmap.ifft(fftpack.ifftshift(fft_deconvolved_cmb))), cmb_wcs)
    enmap_deconvolved_cmb = np.real(enmap.ifft(fftpack.ifftshift(fft_deconvolved_cmb)))

    return enmap_deconvolved_cmb

# specified dec, ra will be at center of map (in between pixels)
def get_deconvolved_map_even(hp_map, beam_Bl, decimal_dec, decimal_ra, cmb_radius_deg, res, lmax, proj):
    """Extracts deconvolved CMB map with center (in between pixels) at specifed dec, ra. 

    Args:
        hp_map (map): Healpix map from healpy
        beam_Bl (list of float): [Bl(l=0), Bl(l=1),.... Bl(l=lmax)]
        decimal_dec (float): Declination in decimal degrees
        decimal_ra (float): Right ascension in decimal degrees
        cmb_radius_deg (float): CMB map radius in decimal degrees. Must be so that the resulting map is odd.
        res (float): Resolution of map (pixel size) in arcminutes
        lmax (int): lmax of deconvolved map

    Raises:
        Exception: cmb_radius_deg must result in odd shape map (for deconvolution)
        Exception: Resulting map will have even shape

    Returns:
        Pixell ndmap: Deconvolved CMB map with no WCS.
    """
    # we need to add half a pixel arcsecond offset (left and up), then cut afterwards to match even shape (no center pixel)
    # RA and DEC are increasing left and up

    # res in arcminutes
    half_res_deg = res / 2 / 60
    enmap_deconvolved_cmb = get_deconvolved_map_odd(hp_map, beam_Bl, decimal_dec + half_res_deg, decimal_ra + half_res_deg, cmb_radius_deg, res=res, lmax=lmax, proj=proj)
    floor_shape = enmap_deconvolved_cmb.shape[0] // 2
    center_pix = (enmap_deconvolved_cmb.shape[0] - 1) / 2
    return enmap_deconvolved_cmb[int(center_pix - (floor_shape - 1)): int(center_pix + (floor_shape + 1)), int(center_pix - (floor_shape - 1)): int(center_pix + (floor_shape + 1))]

def get_deconvolved_map(oshape, hp_map, beam_Bl, decimal_dec, decimal_ra, cmb_radius_deg, res, lmax, proj):
    assert oshape[0] == oshape[1], "I don't think we will (or can) ever use mismatched axis lengths, so this must be a mistake."
    if oshape[0] % 2 == 0:
        enmap_deconvolved_cmb = get_deconvolved_map_even(hp_map, beam_Bl, decimal_dec, decimal_ra, cmb_radius_deg, res=res, lmax=lmax, proj=proj)
    else:
        enmap_deconvolved_cmb = get_deconvolved_map_odd(hp_map, beam_Bl, decimal_dec, decimal_ra, cmb_radius_deg, res=res, lmax=lmax, proj=proj)
    
    crop_amount = (enmap_deconvolved_cmb.shape[0] - oshape[0]) / 2
    assert int(crop_amount) == crop_amount, "If this trips, get_deconvolved_map is not correct in even/odd"
    crop_amount = int(crop_amount)
    cutout = enmap_deconvolved_cmb[crop_amount:-crop_amount, crop_amount:-crop_amount]
    return cutout