import numpy as np
from pixell import enmap, reproject, utils

import matplotlib.pyplot as plt
from matplotlib import cm

import beam_utils
import plot_utils

from extract_maps import extract_maps

def fit_map(fpath_dict, beam_map_width,
                dec, ra, map_radius, R500, parinfo, fit_func,
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
        parinfo (list of dictionaries): parinfo as specified in mpfit docs 
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
    sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150 = extract_maps(fpath_dict, beam_map_width,
                dec, ra, map_radius,
                show_map_plots=False, verbose=False)

    excise_regions = None #[(14, 0, 8, 7)]
    if verbose:
        print('Running simultaneous fit...')
    '''m = mpfit_spherical_gNFW.mpfit_3dgnfw_simultaneous(R500, beam_handler_150, beam_handler_90, sfl_150,
                                                    sfl_90, err_150, err_90, parinfo, excise_regions, num_processes)'''
    m = fit_func(R500, beam_handler_150, beam_handler_90, sfl_150,
                        sfl_90, err_150, err_90, parinfo, excise_regions, num_processes)

    if verbose:
        print('fit params:', m.params)
        print('fit error:', m.perror)
        print('signal to noise ratios:', abs(m.params / m.perror))

    if show_map_plots:
        # will error if we used spherical model (bookkeeping for future (will we just use ellipsoid always?))
        import ellipsoid_model as ellipsoid_model
        theta, P0_150, P0_90, r_x, r_y, r_z, x_offset, y_offset, c_150, c_90 = m.params
        fit_150 = ellipsoid_model.eval_pixel_centers(theta, P0_150, r_x, r_y, r_z, 10, R500, x_offset, y_offset, sfl_150.shape[0]*3, sfl_150.shape[1]*3)
        # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
        fit_150 = ellipsoid_model.rebin_2d(fit_150, (3, 3))
        fit_90 = fit_150 * (P0_90/P0_150)
        plt.figure(0)
        plt.title('sfl 150 w/ gaussian blur')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_150, -100, 100)
        plt.figure(1)
        plt.title('sfl 150 fit')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, fit_150 + c_150, -100, 100)
        plt.figure(2)
        plt.title('sfl 90 w/ gaussian blur')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, sfl_90, -100, 100)
        plt.figure(3)
        plt.title('sfl 90 fit')
        plot_utils.imshow_gaussian_blur_default(1.5, 1.5, fit_90 + c_90, -100, 100)
        plt.show()
    return m.params, m.perror



    
