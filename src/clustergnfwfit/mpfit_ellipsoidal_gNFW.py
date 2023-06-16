import numpy as np

from mpfit import mpfit
import ellipsoid_model as ellipsoid_model

# Fix code later, make less messy
def myfunctgnfw_simul(p, fjac=None, R500=None, y150=None, y90=None, err150=None, err90=None, beam_handler_150=None, beam_handler_90=None, bolocam_map=None, bolocam_err=None, beam_handler_bolocam=None, excise_regions=None):
    """Function to be minimized. Returns weighted deviations between model and data.

    Args:
        p (tuple): parameter values passed in by mpfit
        fjac (boolean, optional): If fjac==None then partial derivatives
        should not be computed.  It will always be None if MPFIT is called
        with default flag. Defaults to None.
        R500 (float, optional): R500 passed in through functkw. Defaults to None.
        y150 (2d array, optional): y150 passed in through functkw. Defaults to None.
        y90 (2d array, optional): ... Defaults to None.
        err150 (2d array, optional): ... Defaults to None.
        err90 (2d array, optional): ... Defaults to None.
        beam_handler_150 (beam_utils.BeamHandler, optional): ... Defaults to None.
        beam_handler_90 (beam_utils.BeamHandler, optional): ... Defaults to None.
        excise_regions (array of tuples, optional): ... Defaults to None.
        num_processes (int, optional): ... Defaults to None

    Returns:
        1d array: relevant weighted deviations between model and data.
    """

    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.

    theta, P0_150, P0_90, r_x, r_y, r_z, x_offset, y_offset, c_150, c_90, P0_bolocam, c_bolocam = p
    gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

    psf_padding_act = beam_handler_150.get_pad_pixels()
    # can use this to make the 90 model beause only P0 is different
    model_act_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, x_offset, y_offset,
                        (y90.shape[0] + psf_padding_act)*3, (y90.shape[1] + psf_padding_act)*3)
    # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
    model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

    model_150_no_c = model_act_no_c * P0_150
    model_90_no_c = model_act_no_c * P0_90

    model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
    model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)

    if beam_handler_bolocam is not None:
        psf_padding_bolocam = beam_handler_bolocam.get_pad_pixels()
        # eval bolocam at 5 arcsecond res, rebin to 20
        model_bolo_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, x_offset, y_offset,
                                                                        (bolocam_map.shape[0] + psf_padding_bolocam)*2, (bolocam_map.shape[1] + psf_padding_bolocam)*2)
        model_bolo_no_c = ellipsoid_model.rebin_2d(model_bolo_no_c, (2, 2))
        # use 150 to make bolocam model because only P0 is different
        model_bolo_no_c = model_bolo_no_c * P0_bolocam

        model_bolocam = beam_handler_bolocam.convolve2d(model_bolo_no_c + c_bolocam, cut_padding=True)



    if excise_regions is None:
        deviation150 = (y150.flatten() - model_150.flatten()) / err150.flatten()
        deviation90 = (y90.flatten() - model_90.flatten()) / err90.flatten()
        if beam_handler_bolocam is not None:
            deviation_bolocam = (bolocam_map.flatten() - model_bolocam.flatten()) / bolocam_err.flatten()
        else:
            deviation_bolocam = np.array([])
    else:
        # mark excised region indices
        for region in excise_regions:
            x, y, width, height = region
            model_150[y:y+height, x:x+width] = np.NaN
            model_90[y:y+height, x:x+width] = np.NaN

        # flatten copies the array; don't have to worry about messing up data that's passed in
        y150 = y150.flatten()
        y90 = y90.flatten()
        model_150 = model_150.flatten()
        model_90 = model_90.flatten()
        err150 = err150.flatten()
        err90 = err90.flatten()

        # get nan indices (to exclude) in flattened arrays
        model_150_nan = np.argwhere(np.isnan(model_150))
        model_90_nan = np.argwhere(np.isnan(model_90))

        # exclude excised region
        y150 = np.delete(y150, model_150_nan)
        y90 = np.delete(y90, model_90_nan)
        model_150 = np.delete(model_150, model_150_nan)
        model_90 = np.delete(model_90, model_90_nan)
        err150 = np.delete(err150, model_150_nan)
        err90 = np.delete(err90, model_90_nan)

        deviation150 = (y150 - model_150) / err150
        deviation90 = (y90 - model_90) / err90

    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    # print('model', model)
    # print('y', y)
    return [status, np.concatenate((deviation150, deviation90, deviation_bolocam))]


def mpfit_ellipsoidal_simultaneous(R500, beam_handler_150, beam_handler_90, y150, y90, err150, err90, parinfo,
                                   bolocam_map=None, bolocam_err=None, beam_handler_bolocam=None, excise_regions=None):
    """Uses mpfit to simultaneously fit 2 maps to one gNFW model that only differs with different P0s and cs for each.

    Args:
        R500 (float): R500 (arcseconds) of the cluster being analyzed. All values outside
        of 5*R500 will be ignored when computing the gNFW model.
        beam_handler_150 (beam_utils.BeamHandler): will be used for beam convolution for 150 GHz variant of model.
        beam_handler_90 (beam_utils.BeamHandler): will be used for beam convolution for 90 GHz variant of model.
        y150 (2d array): contains 150 GHz brightness values. (microKelvins)
        y90 (2d array): contains 90 GHz brightness values. (microKelvins)
        err150 (2d array): contains one-sigma errors corresponding to the values of y150. (microKelvins)
        err90 (2d array): contains one-sigma errors corresponding to the values of y90. (microKelvins)
        init_params (iterable): (P0_150, P0_90, RS (arcseconds), x_offset (pixels), y_offset (pixels), c_150, c_90)
        fixed_params (list of dictionaries): parinfo as specified in mpfit docs 
        excise_regions (list, optional): list of tuples in form of (x, y, width, height)
        representing rectangular regions to be excluded. Coordinate system
        is (0, 0) top left going positive to the down and right. Defaults to None.
        num_processes (int): Max number of cores to use

    Returns:
        mpfit: An object of type mpfit Results are attributes of this class.
        e.g. mpfit.status, mpfit.errmsg, mpfit.params, mpfit.perror, mpfit.niter, mpfit.covar.
        .status
    """
    
    fa = {'R500': R500, 'y150': y150, 'y90': y90, 'err150': err150, 'err90': err90,
          'beam_handler_150': beam_handler_150, 'beam_handler_90': beam_handler_90,
          'bolocam_map': bolocam_map, 'bolocam_err': bolocam_err, 'beam_handler_bolocam': beam_handler_bolocam, 
          'excise_regions': excise_regions,}
    m = mpfit(myfunctgnfw_simul, parinfo=parinfo, functkw=fa)

    return m

