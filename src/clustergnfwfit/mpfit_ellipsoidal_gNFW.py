import numpy as np

from mpfit import mpfit
import ellipsoid_model as ellipsoid_model

# Fix code later, make less messy
def myfunctgnfw_simul(p, fjac=None, R500=None, data_150=None, data_90=None, err_150=None, err_90=None, beam_handler_150=None,
                      beam_handler_90=None, data_bolocam=None, err_bolocam=None, beam_handler_bolocam=None,
                      data_milca=None, err_milca=None, beam_handler_milca=None,
                      use_act=None, use_bolocam=None, use_milca=None):
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
        num_processes (int, optional): ... Defaults to None

    Returns:
        1d array: relevant weighted deviations between model and data.
    """

    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.

    theta, r_x, r_y, r_z, offset_x, offset_y = p[:6]
    p = p[6:]

    if use_act:
        p0_150, p0_90, c_150, c_90 = p[:4]
        p = p[4:]
    
    if use_bolocam: 
        p0_bolocam, c_bolocam = p[:2]
        p = p[2:]
    
    if use_milca:
        p0_milca = p[0]

    gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

    if (use_act is False or data_90 is None or data_90.shape[0] % 2 == 0) and (use_act is False or data_bolocam is None or data_bolocam.shape[0] % 2 == 0) and (use_milca is False or data_milca is None or data_milca.shape[0] % 2 == 0):
        # can use even shape of both act and bolocam data to eval the model map only once, then rebin -> speed up
        # evaluate the bigger map

        # act and bolocam are rebinned, then convolved
        # milca is convolved without padding (so also without cutting off padding), then rebinned
        map_size = 0
        if use_act:
            act_map_size = (data_90.shape[0] + beam_handler_90.get_pad_pixels())*3
            map_size = max(map_size, act_map_size)
        if use_bolocam:
            bolocam_map_size = (data_bolocam.shape[0] + beam_handler_bolocam.get_pad_pixels())*2
            map_size = max(map_size, bolocam_map_size)
        if use_milca:
            milca_map_size = (data_milca.shape[0]) * 20
            map_size = max(map_size, milca_map_size)
        model_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                                                        map_size, map_size)
        
        if use_act:
            act_crop_amount = (map_size - act_map_size) / 2
            assert int(act_crop_amount) == act_crop_amount
            act_crop_amount = int(act_crop_amount)
            if act_crop_amount > 0:
                model_act_no_c = model_no_c[act_crop_amount:-act_crop_amount, act_crop_amount:-act_crop_amount]
            else:
                model_act_no_c = model_no_c
            model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

            model_90_no_c = model_act_no_c * p0_90
            model_150_no_c = model_act_no_c * p0_150

            model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
            model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)

        if use_bolocam:
            bolo_crop_amount = (map_size - bolocam_map_size) / 2
            assert int(bolo_crop_amount) == bolo_crop_amount
            bolo_crop_amount = int(bolo_crop_amount)
            if bolo_crop_amount > 0:
                model_bolo_no_c = model_no_c[bolo_crop_amount:-bolo_crop_amount, bolo_crop_amount:-bolo_crop_amount]
            else:
                model_bolo_no_c = model_no_c
            model_bolo_no_c = ellipsoid_model.rebin_2d(model_bolo_no_c, (2, 2))

            model_bolo_no_c = model_bolo_no_c * p0_bolocam

            model_bolocam = beam_handler_bolocam.convolve2d(model_bolo_no_c + c_bolocam, cut_padding=True)

        if use_milca:
            milca_crop_amount = (map_size - milca_map_size) / 2
            assert int(milca_crop_amount) == milca_crop_amount
            milca_crop_amount = int(milca_crop_amount)
            if milca_crop_amount > 0:
                model_milca_no_c = model_no_c[milca_crop_amount:-milca_crop_amount, milca_crop_amount:-milca_crop_amount]
            else:
                model_milca_no_c = model_no_c
            model_milca = beam_handler_milca.convolve2d(model_milca_no_c, cut_padding=False)
            model_milca = ellipsoid_model.rebin_2d(model_milca, (20, 20))

            model_milca = model_milca * p0_milca

    else:
        if use_act:
            psf_padding_act = beam_handler_150.get_pad_pixels()
            # can use this to make the 90 model beause only P0 is different
            model_act_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                (data_90.shape[0] + psf_padding_act)*3, (data_90.shape[1] + psf_padding_act)*3)
            # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
            model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

            model_90_no_c = model_act_no_c * p0_90
            model_150_no_c = model_act_no_c * p0_150

            model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
            model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)

        if use_bolocam:
            psf_padding_bolocam = beam_handler_bolocam.get_pad_pixels()
            # eval bolocam at 5 arcsecond res, rebin to 20
            model_bolo_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                                                            (data_bolocam.shape[0] + psf_padding_bolocam)*2, (data_bolocam.shape[1] + psf_padding_bolocam)*2)
            # evaluated at 10 arcsecond resolution, rebin to 20 arcsecond pixels
            model_bolo_no_c = ellipsoid_model.rebin_2d(model_bolo_no_c, (2, 2))

            model_bolo_no_c = model_bolo_no_c * p0_bolocam

            model_bolocam = beam_handler_bolocam.convolve2d(model_bolo_no_c + c_bolocam, cut_padding=True)

        if use_milca:
            model_milca_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                                                            (data_milca.shape[0])*20, (data_milca.shape[1])*20)
            model_milca = beam_handler_milca.convolve2d(model_milca_no_c, cut_padding=False)
            model_milca = ellipsoid_model.rebin_2d(model_milca, (20, 20))

            model_milca = model_milca * p0_milca


    if use_act:
        deviation_150 = (data_150.flatten() - model_150.flatten()) / err_150.flatten()
        deviation_90 = (data_90.flatten() - model_90.flatten()) / err_90.flatten()
    else:
        deviation_90 = np.array([])
        deviation_150 = np.array([])

    if use_bolocam:
        deviation_bolocam = (data_bolocam.flatten() - model_bolocam.flatten()) / err_bolocam.flatten()
    else:
        deviation_bolocam = np.array([])

    if use_milca:
        deviation_milca = (data_milca.flatten() - model_milca.flatten()) / err_milca.flatten()
    else:
        deviation_milca = np.array([])

    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    # print('model', model)
    # print('y', y)
    return [status, np.concatenate((deviation_150, deviation_90, deviation_bolocam, deviation_milca))]


def mpfit_ellipsoidal_simultaneous(parinfo, R500, beam_handler_150=None, beam_handler_90=None, data_150=None, data_90=None, err_150=None, err_90=None,
                                   data_bolocam=None, err_bolocam=None, beam_handler_bolocam=None, data_milca=None, err_milca=None, beam_handler_milca=None,
                                   use_act=True, use_bolocam=True, use_milca=True):
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
        representing rectangular regions to be excluded. Coordinate system
        is (0, 0) top left going positive to the down and right. Defaults to None.
        num_processes (int): Max number of cores to use

    Returns:
        mpfit: An object of type mpfit Results are attributes of this class.
        e.g. mpfit.status, mpfit.errmsg, mpfit.params, mpfit.perror, mpfit.niter, mpfit.covar.
        .status
    """
    
    fa = {'R500': R500, 'data_150': data_150, 'data_90': data_90, 'err_150': err_150, 'err_90': err_90,
          'beam_handler_150': beam_handler_150, 'beam_handler_90': beam_handler_90,
          'data_bolocam': data_bolocam, 'err_bolocam': err_bolocam, 'beam_handler_bolocam': beam_handler_bolocam,
          'data_milca': data_milca, 'err_milca': err_milca, 'beam_handler_milca': beam_handler_milca,
          'use_act': use_act, 'use_bolocam': use_bolocam, 'use_milca': use_milca}

    m = mpfit(myfunctgnfw_simul, parinfo=parinfo, functkw=fa)

    return m

