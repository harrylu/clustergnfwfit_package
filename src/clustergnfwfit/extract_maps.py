import numpy as np
from pixell import enmap, reproject, utils
import healpy as hp
from astropy.io import fits
from scipy import fft
from astropy.modeling.functional_models import Gaussian2D

import beam_utils
import deconvolution

def extract_maps(fpath_dict,
                dec, ra, map_radius,
                deconvolve_cmb_lmax=2000, include_bolocam=True, verbose=False):
    """Extracts specified region in ACTPlanck data with cmb subtracted.
    ACTPlanck maps are here: https://lambda.gsfc.nasa.gov/product/act/actpol_dr5_coadd_maps_get.html
    Cmb deconvolution is done via division in Fourier space with low pass filtering.
    Cmb maps are here: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/previews/COM_CMB_IQU-commander_2048_R3.00_full/
    Uses I_STOKES_INP ^.
    Bolocam maps are here: https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/bolocam/bolocam.html

    Args:
        fpath_dict (dict): contains file paths; should contain keys:
            'brightness_150': path to brightness 150 fits file,
            'noise_150': 'path to 150 GHz noise fits file,
            'brightness_90': path to 90 GHz brightness fits file,
            'noise_90', 'path to 90 GHz noise fits file,
            'cmb', 'beam_150', 'beam_90': self-explanatory
            Optional:
            'bolocam_filtered': path to filtered bolocam fits file,
            'bolocam_noise': path to bolocam RMS noise fits file,
            'bolocam_transfer': path to bolocam transfer fits file,
        dec (tuple): declination in (degrees, minutes, seconds)
        ra (tuple): right ascension in (hours, minutes, seconds)
        map_radius (float): in arcminutes; radial width of the map that will be extracted
        deconvolve_cmb_lmax (int): lmax to keep in deconvolved cmb 
        verbose (bool, optional): Whether to log to console. Defaults to False.

    Notes:
        The extracted maps will be centered at the (dec, ra) and so will always have to be an odd-numbered shape.
        Will raise error otherwise.

    Returns:
        6 length tuple
        Elements 0, 1: SFL-reprojected maps of specified map_radius 90 GHz, 150 GHz, respectively
        Elements 2, 3: One sigma errors on the maps, 90 GHz, 150 GHz respectively
        Elements 4, 5: BeamHandler instances 90 GHz, 150 GHz, respectively
        Elements 6, 7: bolocam_map, bolocam_beam_handler or None, None if include_bolocam=False


    """
    
    def hms_to_deg(hours, minutes, seconds):
        return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
    def dms_to_deg(degrees, minutes, seconds):
        return degrees + minutes / 60 + seconds / (60 ** 2)

    decimal_dec = dms_to_deg(*dec)
    decimal_ra = hms_to_deg(*ra)
    

    # x arcmins = x/60 deg
    # add some positive arcmin to map_radius so we dont lose data when we reproject
    deg_r = (map_radius + .5) / 60

    # Create the box and use it to select a submap enmap
    box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

    # resolution is 30 arcseconds, will use later
    res = 1/2

    # these are in CAR projection (may not be centered on dec, ra)
    enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
    enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
    enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
    enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
    
    radius = map_radius*utils.arcmin
    even_maps = True
    # Need to do some stuff if we want even maps
    if even_maps:
        half_pixel_deg = res / 60 / 2
        coords = [np.deg2rad([decimal_dec + half_pixel_deg, decimal_ra + half_pixel_deg])]
        sfl_90 = reproject.thumbnails(enmap_90, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150 = reproject.thumbnails(enmap_150, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_90 = sfl_90[1:, 1:]
        sfl_90_noise = sfl_90_noise[1:, 1:]
        sfl_150 = sfl_150[1:, 1:]
        sfl_150_noise = sfl_150_noise[1:, 1:]

    else:
        # for odd maps
        # reproject to sfl thumbnails (definitely centered on dec, ra)
        coords = [np.deg2rad([decimal_dec, decimal_ra])]
        sfl_90 = reproject.thumbnails(enmap_90, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150 = reproject.thumbnails(enmap_150, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=res * utils.arcmin, proj='sfl', verbose=verbose)[0]
        print(f"ACTPlanck SFL WCS: {sfl_90.wcs}")
    

    assert sfl_90.shape[0] == sfl_90.shape[1], f"Sfl 90 axis length mismatch: {sfl_90.shape}"
    assert sfl_90.shape == sfl_150.shape

    # I_STOKES_INP is column (field) 5
    hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
    # extract 1 degree x 1 degree map (slightly larger so that there is center pixel)
    cmb_radius_deg = 0.503
    # get CMB beam
    hdul = fits.open(fpath_dict['cmb'])
    beam_hdu = hdul[2]
    Bl = list(beam_hdu.columns['INT_BEAM'].array)

    # diameter of 17 pixels has pixels at < 1% of highest
    beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], 17)
    beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], 17)

    enmap_deconvolved_cmb = deconvolution.get_deconvolved_map(np.array(sfl_90.shape) + beam_handler_90.get_pad_pixels(), hp_map, Bl, decimal_dec, decimal_ra, cmb_radius_deg, res=1/2, lmax=deconvolve_cmb_lmax, proj='sfl')

    # convolve deconvolved cmb with 90 GHz, 150 GHz beams
    deconvolved_cmb_90 = beam_handler_90.convolve2d(enmap_deconvolved_cmb)
    deconvolved_cmb_150 = beam_handler_150.convolve2d(enmap_deconvolved_cmb)

    # subtract cmb
    sfl_90_cmb_subtracted = sfl_90 - deconvolved_cmb_90
    sfl_150_cmb_subtracted = sfl_150 - deconvolved_cmb_150 

    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.figure('deconvolved cmb 90')
    plt.imshow(deconvolved_cmb_90, cmap=cm.coolwarm, vmin=-100, vmax=100)
    plt.show

    def ivar_to_sigma(x): return np.sqrt(1 / x)
    err_90 = ivar_to_sigma(sfl_90_noise)
    err_150 = ivar_to_sigma(sfl_150_noise)

    # get bolocam maps
    if include_bolocam == True:
        #read FITS header
        header = fits.open(fpath_dict['bolocam_filtered'])[0].header

        # wcs incorrect when read in this way
        enmap_bolocam_filtered = enmap.read_fits(fpath_dict['bolocam_filtered'])
        # fix WCS
        enmap_bolocam_filtered.wcs.wcs.cdelt = [float(header['CD1_1']), float(header['CD2_2'])]
        print(f'Bolocam WCS: {enmap_bolocam_filtered.wcs}')

        # beam is approx gaussian, fwhm in degrees
        bolocam_beam_fwhm = header['BMAJ']
        bolocam_beam_handler = beam_utils.BeamHandlerBolocam(bolocam_beam_fwhm, 11)

        # 20 arcsecond resolution
        enmap_deconvolved_cmb = deconvolution.get_deconvolved_map(np.array(enmap_bolocam_filtered.shape) + bolocam_beam_handler.get_pad_pixels(), hp_map, Bl, decimal_dec, decimal_ra, cmb_radius_deg, res=1/3, lmax=deconvolve_cmb_lmax, proj='sfl')
        assert enmap_deconvolved_cmb.shape[0] % 2 == 0
        assert enmap_deconvolved_cmb.shape[0] == enmap_deconvolved_cmb.shape[1]

        # convolve CMB with Bolocam psf
        enmap_deconvolved_cmb = bolocam_beam_handler.convolve2d(enmap_deconvolved_cmb)
        
        # apply hanning
        hanning = np.outer(np.hanning(42), np.hanning(42))
        hanning /= np.mean(hanning)
        enmap_deconvolved_cmb *= hanning

        # filter cmb
        transfer_function_hdul = fits.open(fpath_dict['bolocam_transfer'])
        signal_transfer_function_fft = transfer_function_hdul[0].data + 1j * transfer_function_hdul[1].data
        enmap_deconvolved_cmb = np.real(fft.ifft2(fft.fft2(enmap_deconvolved_cmb) * signal_transfer_function_fft))

        bolocam_map = enmap_bolocam_filtered - enmap_deconvolved_cmb

        bolocam_err = fits.open(fpath_dict['bolocam_noise'])[0].data

        
    else:
        bolocam_map = None
        bolocam_err = None
        bolocam_beam_handler = None

    return sfl_90_cmb_subtracted, sfl_150_cmb_subtracted, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_err, bolocam_beam_handler
