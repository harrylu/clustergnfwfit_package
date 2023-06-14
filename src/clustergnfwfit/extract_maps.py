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
    coords = [np.deg2rad([decimal_dec, decimal_ra])]

    # x arcmins = x/60 deg
    # we add some positive arcmin to the map radius to prevent losing edge data when we reproject (maybe unnecessary?)
    # .77 gives odd size of map
    deg_r = (map_radius + .77) / 60

    # Create the box and use it to select a submap enmap
    box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

    # resolution is 30 arcseconds, will use later
    res = 1/2 * utils.arcmin

    # these are in CAR projection
    enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
    enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
    enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
    enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
    # want odd shape for center pixel
    if (enmap_90.shape[0] % 2 == 0 or enmap_90.shape[1] % 2 == 0):
        raise Exception(f"Tweak map_radius (Trial and error; try values close to {map_radius}). Resulting map shape should be odd (for subtracting deconvolved cmb) instead of {enmap_90.shape}.")

    # I_STOKES_INP is column (field) 5
    hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
    # extract 1 degree x 1 degree map (slightly larger so that there is center pixel)
    cmb_radius_deg = 0.503
    # get CMB beam
    hdul = fits.open(fpath_dict['cmb'])
    beam_hdu = hdul[2]
    Bl = list(beam_hdu.columns['INT_BEAM'].array)
    enmap_deconvolved_cmb = deconvolution.get_deconvolved_map_fft(hp_map, Bl, decimal_dec, decimal_ra, cmb_radius_deg, res=1/2, lmax=deconvolve_cmb_lmax)

    # diameter of 17 pixels has pixels at < 1% of highest
    beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], 17)
    beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], 17)

    # convolve deconvolved cmb with 90 GHz, 150 GHz beams
    deconvolved_cmb_90 = beam_handler_90.convolve2d(enmap_deconvolved_cmb)
    deconvolved_cmb_150 = beam_handler_150.convolve2d(enmap_deconvolved_cmb)

    # cut out region of cmb for subtraction
    center_pix_y = deconvolved_cmb_90.shape[0] // 2
    center_pix_x = deconvolved_cmb_90.shape[1] // 2
    cut_amount = enmap_90.shape[0] // 2
    deconvolved_cmb_cutout_90 = deconvolved_cmb_90[center_pix_y - cut_amount:center_pix_y + cut_amount + 1, center_pix_x - cut_amount:center_pix_x + cut_amount + 1]
    deconvolved_cmb_cutout_150 = deconvolved_cmb_150[center_pix_y - cut_amount:center_pix_y + cut_amount + 1, center_pix_x - cut_amount:center_pix_x + cut_amount + 1]

    # subtract cmb
    enmap_90_cmb_subtracted = enmap_90 - deconvolved_cmb_cutout_90
    enmap_150_cmb_subtracted = enmap_150 - deconvolved_cmb_cutout_150 

    # reproject to sfl
    radius = map_radius*utils.arcmin
    sfl_90 = reproject.thumbnails(enmap_90_cmb_subtracted, coords, r=radius, res=res, proj='sfl', verbose=verbose)[0]
    sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=res, proj='sfl', verbose=verbose)[0]
    sfl_150 = reproject.thumbnails(enmap_150_cmb_subtracted, coords, r=radius, res=res, proj='sfl', verbose=verbose)[0]
    sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=res, proj='sfl', verbose=verbose)[0]

    def ivar_to_sigma(x): return np.sqrt(1 / x)
    err_90 = ivar_to_sigma(sfl_90_noise)
    err_150 = ivar_to_sigma(sfl_150_noise)

    # get bolocam maps
    if include_bolocam == True:
        # 20 arcsecond resolution
        # we need to add 10 arcsecond offset, then cut afterwards to match bolocam's 42 x 42 pixels (no center pixel)
        ten_arcseconds_deg = 0.00277778
        enmap_deconvolved_cmb = deconvolution.get_deconvolved_map_fft(hp_map, Bl, decimal_dec + ten_arcseconds_deg, decimal_ra + ten_arcseconds_deg, cmb_radius_deg, res=1/3, lmax=deconvolve_cmb_lmax)

        # convolve with Bolocam psf
        header = fits.open(fpath_dict['filtered'])[0].header
        # beam is approx gaussian, fwhm in degrees
        bolocam_beam_fwhm = header['BMAJ']

        bolocam_beam_handler = beam_utils.BeamHandlerBolocam(bolocam_beam_fwhm, 11)

        sfl_cmb = reproject.thumbnails(enmap_deconvolved_cmb, coords, r=cmb_radius_deg*60*utils.arcmin, res=1/3*utils.arcmin, proj='sfl')[0]
        sfl_cmb = bolocam_beam_handler.convolve2d(sfl_cmb)
        center_pix = np.array(sfl_cmb.shape) // 2
        sfl_cmb = sfl_cmb[center_pix[0]-20:center_pix[0]+22, center_pix[1]-20:center_pix[1]+22]


        # apply hanning
        hanning = np.outer(np.hanning(42), np.hanning(42))
        hanning /= np.mean(hanning)
        sfl_cmb *= hanning

        # filter cmb
        transfer_function_hdul = fits.open(fpath_dict['transfer'])
        signal_transfer_function_fft = transfer_function_hdul[0].data + 1j * transfer_function_hdul[1].data
        sfl_cmb = np.real(fft.ifft2(fft.fft2(sfl_cmb) * signal_transfer_function_fft))

        enmap_bolocam_filtered = enmap.read_fits(fpath_dict['filtered'])
        # wcs likely incorrect when read in this way so dont use the wcs

        bolocam_map = enmap_bolocam_filtered - sfl_cmb
    else:
        bolocam_map = None
        bolocam_beam_handler = None

    return sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_beam_handler
