import numpy as np
from pixell import enmap, reproject, utils

import matplotlib.pyplot as plt
from matplotlib import cm

from . import beam_utils
from . import plot_utils
from . import mpfit_gNFW

def fit_map(fpath_dict, beam_map_width,
                dec, ra, map_radius, R500, init_params, fixed_params,
                show_map_plots=False, verbose=False, num_processes=4):
    """Runs mpfit on the specified map

    Args:
        fpath_dict (dict): contains file paths; should contain keys:
        'brightness_150': path to brightness 150 fits file,
        'noise_150': 'path to 150 GHz noise fits file,
        'brightness_90': path to 90 GHz brightness fits file,
        'noise_90', 'path to 90 GHz noise fits file,
        'cmb', 'beam_150', 'beam_90': self-explanatory
        dec (tuple): declination in (degrees, minutes, seconds)
        ra (tuple): right ascension in (hours, minutes, seconds)
        map_radius (float): in arcminutes; radial width of the map that will be extracted
        R500 (float): R500 value (arcseconds)
        init_params (iterable): (P0_150, P0_90, RS (arcseconds), x_offset (pixels), y_offset (pixels), c_150, c_90)
        fixed_params (iterable of booleans): Whether each parameter is fixed 
        show_map_plots (bool, optional): Whether to show matplotlib plots. Defaults to False.
        verbose (bool, optional): Whether to log to console. Defaults to False.
        num_processes (int, optional): Max number of cores to use. Defaults to 4.

    Notes:
        The extracted maps will be centered at the (dec, ra) and so will always be an odd-numbered shape.

        I first read in a submap and then reproject, so it is possible that the border might be a bit off. In the future,
        it might be good to read in a little more than neccessary to counteract that.

        It's possible that the cmb reprojection is not at the correct coordinates. Look into this.

    Returns:
        1. tuple: (P0_150, P0_90, RS, x_offset, y_offset, c_150, c_90) - the parameters of the fit
        The units of the tuple are ( , , RS: arcseconds, x_offset: pixels, y_offset: pixels, , )
        2. tuple: one sigma error for each of the parameters

    """
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
    # we add 1.5 arcmin (3 pixel) to the map radius to prevent losing edge data when we reproject (maybe unnecessary?)
    deg_r = (map_radius + 1.5) / 60

    # Create the box and use it to select a submap enmap
    box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

    # these are in CAR projection
    enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
    enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
    enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
    enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
    # after lots of effort, I discovered that reproject.enmap_from_healpix
    # is only correct if the wcs is CAR (or maybe just original wcs from FITS file)
    # and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen
    enmap_cmb = reproject.enmap_from_healpix(fpath_dict['cmb'], enmap_150.shape, enmap_150.wcs, 
                                        ncomp=1, unit=1e-6, lmax=6000,rot='gal,equ')[0]
    
    # subtract the cmb from the actplanck maps
    enmap_150 -= enmap_cmb
    enmap_90 -= enmap_cmb

    # should we deconvolve the thumbnails?
    # after lots of trouble, finally realized that res parameter was short for resolution or something
    # we can set a resolution of 1/2 * utils.arcmin (30 arcseconds)!!!
    radius = map_radius*utils.arcmin
    resolution = 1/2 * utils.arcmin
    sfl_150 = reproject.thumbnails(enmap_150, coords, r=radius, res=resolution, proj='sfl', verbose=verbose)[0]
    sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=resolution, proj='sfl', verbose=verbose)[0]
    sfl_90 = reproject.thumbnails(enmap_90, coords, r=radius, res=resolution, proj='sfl', verbose=verbose)[0]
    sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=resolution, proj='sfl', verbose=verbose)[0]
    
    # reprojection flips the maps for some reason; need to flip back
    sfl_150 = np.flip(sfl_150, 1)
    sfl_150_noise = np.flip(sfl_150_noise, 1)
    sfl_90 = np.flip(sfl_90, 1)
    sfl_90_noise = np.flip(sfl_90_noise, 1)

    # need to convert noise inverse variance to sigma for ivar maps
    def ivar_to_sigma(x): return np.sqrt(1 / x)
    err_150 = ivar_to_sigma(sfl_150_noise)
    err_90 = ivar_to_sigma(sfl_90_noise)

    if show_map_plots:
        plt.figure(0)
        plt.title('sfl 150 w/ gaussian blur')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_150, -100, 100)
        plt.figure(1)
        plt.title('sfl 150 noise')
        plt.imshow(err_150, cmap=cm.coolwarm, vmin=-100, vmax=100)
        plt.figure(2)
        plt.title('sfl 90 w/ gaussian blur')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_90, -100, 100)
        plt.figure(3)
        plt.title('sfl 90 noise')
        plt.imshow(err_90, cmap=cm.coolwarm, vmin=-100, vmax=100)
        plt.figure(4)
        plt.title('cmb')
        plt.imshow(enmap_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)
        plt.show()

    if verbose:
        print('Instantiating beam handlers')
    beam_handler_150 = beam_utils.BeamHandler2D(fpath_dict['beam_150'], beam_map_width)
    beam_handler_90 = beam_utils.BeamHandler2D(fpath_dict['beam_90'], beam_map_width)

    excise_regions = None #[(14, 0, 8, 7)]
    if verbose:
        print('Running simultaneous fit...')
    m = mpfit_gNFW.mpfit_3dgnfw_simultaneous(R500, beam_handler_150, beam_handler_90, sfl_150,
                                                    sfl_90, err_150, err_90, init_params, fixed_params, excise_regions, num_processes)

    if verbose:
        print('fit params:', m.params)
        print('fit error:', m.perror)
        print('signal to noise ratios:', abs(m.params / m.perror))

    return m.params, m.perror



    
