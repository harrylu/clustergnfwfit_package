import numpy as np
from scipy import fftpack
from pixell import enmap, reproject, utils
from astropy.io import fits
from astropy.units import cds

import gc

import beam_utils
import deconvolution

# specified 
def get_deconvolved_map_odd(ndmap_cmb, beam_Bl, coords, cmb_radius_deg, res, lmax, proj):
    """Extracts deconvolved CMB maps with center pixel at specifed dec, ra. 

    Args:
        ndmap_cmb (map): ndmap of cmb, must include specified region
        beam_Bl (list of float): [Bl(l=0), Bl(l=1),.... Bl(l=lmax)]
        coords: list of tuples (dec, ra) in rads
        decimal_dec (float): Declination in decimal degrees
        decimal_ra (float): Right ascension in decimal degrees
        cmb_radius_deg (float): CMB map radius in decimal degrees. Must be so that the resulting map is odd.
        res (float): Resolution of map (pixel size) in arcminutes
        lmax (int): lmax of deconvolved map

    Raises:
        Exception: cmb_radius_deg must result in odd shape map (for deconvolution)
        Exception: Resulting map will have odd shape

    Returns:
        List of Pixell ndmaps: Deconvolved CMB map.
    """
    '''
    # reproject from healpix to CAR
    # and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen (maybe not anymore?)
    box_radius_deg = cmb_radius_deg * 1.1    # read in a little extra so reprojection doesnt lose data
    box = np.deg2rad([[decimal_dec - box_radius_deg, decimal_ra - box_radius_deg], [decimal_dec + box_radius_deg, decimal_ra + box_radius_deg]])
    # enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape reads +x, +y
    # so we have to make our own wcs that is 1deg x 1deg and centered at our dec, r
    cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=res * utils.arcmin, proj='car')
    enmap_cmb = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
                                            ncomp=1, unit=1e-6, rot='gal,equ')[0]
    '''
    
    # specified a proj, reproject to that coordinate system
    coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = coords[np.newaxis]

    shape = int(cmb_radius_deg.to(cds.degree).value / res.to(cds.degree).value) * 2
    print(f'Reprojecting CMB to {proj}')
    oshape = (shape, shape)
    oshape, owcs = enmap.thumbnail_geometry(shape=oshape, res=res.to(cds.arcmin).value * utils.arcmin, proj=proj)
    ndmap_cmb = reproject.thumbnails(ndmap_cmb, coords, r=cmb_radius_deg.to(cds.deg).value*utils.degree, oshape=oshape, owcs=owcs, res=res.to(cds.arcmin).value*utils.arcmin, proj=proj, verbose=True)
    print(f"Reprojected oshape: {ndmap_cmb.shape}")
        

    # want odd shape in order to have center pixel
    if (ndmap_cmb.shape[-1] % 2 == 0 or ndmap_cmb.shape[-2] % 2 == 0):
        raise Exception(f"Map should be odd. Weird, should never be even.")


    cmb_beam_handler = beam_utils.BeamHandlerPlanckCMB(beam_Bl, ndmap_cmb.shape[-1])
    cmb_beam_map = cmb_beam_handler.beam_map
    # normalize peak of beam
    cmb_beam_map /= cmb_beam_map[cmb_beam_map.shape[0]//2, cmb_beam_map.shape[1]//2]

    print('fftshift')
    # changed from fftpack to np fft
    fft_ndmap_cmb = np.fft.fftshift(np.array([enmap.fft(m) for m in list(ndmap_cmb)]), axes=(1, 2))
    del ndmap_cmb
    gc.collect()
    fft_beam_cmb = np.fft.fftshift(np.abs(enmap.fft(cmb_beam_map)))

    print('deconvolving through division')
    fft_deconvolved_cmb = fft_ndmap_cmb / fft_beam_cmb[np.newaxis, :, :]
    del fft_ndmap_cmb
    del fft_beam_cmb
    gc.collect()
    # np.divide(fft_ndmap_cmb, fft_beam_cmb)

    # resolution in radians
    # numpy outer_func would do this more elegantly
    rad_res = np.deg2rad(res.to(cds.degree).value)
    y_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[1], rad_res), (fft_deconvolved_cmb.shape[2], 1)).T
    x_freq = np.tile(np.fft.fftfreq(fft_deconvolved_cmb.shape[2], rad_res), (fft_deconvolved_cmb.shape[1], 1))

    # 2pi * freq to convert from k to l
    l_dist = np.fft.fftshift(np.sqrt(np.square(2*np.pi*y_freq) + np.square(2*np.pi*x_freq)))

    # mask out l > lmax
    fft_deconvolved_cmb[:, l_dist > lmax] = 0

    # enmap_deconvolved_cmb = enmap.ndmap(np.real(enmap.ifft(fftpack.ifftshift(fft_deconvolved_cmb))), cmb_wcs)
    print('converting to ndmaps')
    ndmaps = []
    for deconvolved_map_fft in fft_deconvolved_cmb:
        ndmaps.append(enmap.ndmap(np.real(enmap.ifft(np.fft.ifftshift(deconvolved_map_fft))), owcs))
        del deconvolved_map_fft
    gc.collect()
    # ndmaps = [enmap.ndmap(np.real(enmap.ifft(fftpack.ifftshift(deconvolved_map_fft))), owcs) for deconvolved_map_fft in list(fft_deconvolved_cmb)]

    return ndmaps

# specified dec, ra will be at center of map (in between pixels)
def get_deconvolved_map_even(ndmap_cmb, beam_Bl, coords, cmb_radius_deg, res, lmax, proj):
    """Extracts deconvolved CMB map with center (in between pixels) at specifed dec, ra. 

    Args:
        ndmap_cmb (map): ndmap of cmb, must include specified region
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
    half_res_rad = (res / 2).to(cds.rad).value
    ndmaps = deconvolution.get_deconvolved_map_odd(ndmap_cmb, beam_Bl, coords + half_res_rad, cmb_radius_deg, res=res, lmax=lmax, proj=proj)
    
    floor_shape = ndmaps[0].shape[-1] // 2
    center_pix = (ndmaps[0].shape[-1] - 1) / 2

    for i, m in enumerate(ndmaps):
        ndmaps[i] = m[int(center_pix - (floor_shape - 1)): int(center_pix + (floor_shape + 1)), int(center_pix - (floor_shape - 1)): int(center_pix + (floor_shape + 1))]
    return ndmaps

def get_deconvolved_map(oshape, ndmap_cmb, beam_Bl, coords, cmb_radius_deg, res, lmax, proj):
    assert oshape[0] == oshape[1], "I don't think we will (or can) ever use mismatched axis lengths, so this must be a mistake."
    if oshape[0] % 2 == 0:
        ndmaps = deconvolution.get_deconvolved_map_even(ndmap_cmb, beam_Bl, coords, cmb_radius_deg, res=res, lmax=lmax, proj=proj)
    else:
        ndmaps = deconvolution.get_deconvolved_map_odd(ndmap_cmb, beam_Bl, coords, cmb_radius_deg, res=res, lmax=lmax, proj=proj)

    crop_amount = (ndmaps[0].shape[-1] - oshape[-1]) / 2
    assert int(crop_amount) == crop_amount, "If this trips, get_deconvolved_map is not correct in even/odd"
    assert 2 * cmb_radius_deg.to(cds.arcmin).value / res.to(cds.arcmin).value > oshape[-1], f"Increase cmb_radius_deg, {cmb_radius_deg} is too small for oshape of {oshape}. (Largest shape is {cmb_radius_deg.to(cds.arcmin).value / res})"
    crop_amount = int(crop_amount)
    for i, m in enumerate(ndmaps):
        ndmaps[i] = m[crop_amount:-crop_amount, crop_amount:-crop_amount]
    return ndmaps