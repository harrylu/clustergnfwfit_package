import numpy as np
from pixell import enmap, reproject, utils

import matplotlib.pyplot as plt
from matplotlib import cm

import beam_utils
import plot_utils

def extract_maps(fpath_dict, beam_map_width,
                dec, ra, map_radius,
                show_map_plots=False, verbose=False):
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
        show_map_plots (bool, optional): Whether to show matplotlib plots. Defaults to False.
        verbose (bool, optional): Whether to log to console. Defaults to False.

    Notes:
        The extracted maps will be centered at the (dec, ra) and so will always be an odd-numbered shape.

    Returns:
        6 length tuple
        Elements 0, 1: SFL-reprojected maps of specified map_radius 90 GHz, 150 GHz, respectively
        Elements 2, 3: One sigma errors on the maps, 90 GHz, 150 GHz respectively
        Elements 4, 5: BeamHandler instances 90 GHz, 150 GHz, respectively


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
    # we add 1 arcmin (2 pixel) to the map radius to prevent losing edge data when we reproject (maybe unnecessary?)
    deg_r = (map_radius + 1) / 60

    # Create the box and use it to select a submap enmap
    box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

    # these are in CAR projection
    enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
    enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
    enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
    enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
    # use wcs from fits file
    # and also, we cant reproject.thumbnails after enmap_from_healpix or bad things happen
    enmap_cmb = reproject.enmap_from_healpix(fpath_dict['cmb'], enmap_150.shape, enmap_150.wcs, 
                                        ncomp=1, unit=1e-6,rot='gal,equ')[0]
    
    '''plt.figure(-1)
    plt.title('not subtracted 90')
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90, -100, 100)
    plt.figure(-2)
    plt.title('not subtracted 150')
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150, -100, 100)
    plt.figure(4)
    plt.title('cmb')
    plt.imshow(enmap_cmb, cmap=cm.coolwarm, vmin=-100, vmax=100)'''

    
    # subtract the cmb from the actplanck maps
    enmap_150 -= enmap_cmb
    enmap_90 -= enmap_cmb
    
    '''plt.figure(10)
    plt.title('enmap 150 w/ gaussian blur')
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150, -100, 100)
    plt.figure(11)
    plt.title('enmap 90 w/ gaussian blur')
    plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90, -100, 100)
    plt.show()'''

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
        plt.figure(10)
        plt.title('enmap 150 w/ gaussian blur')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_150, -100, 100)
        plt.figure(11)
        plt.title('enmap 90 w/ gaussian blur')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, enmap_90, -100, 100)
        plt.show()
        
    if verbose:
        print('Instantiating beam handlers')
    beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], beam_map_width)
    beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], beam_map_width)

    return sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150