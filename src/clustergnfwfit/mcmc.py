from multiprocessing import Pool
import ellipsoid_model
import mpfit_ellipsoidal_gNFW
from extract_maps import extract_maps
import gnfw_fit_map

import os
import scipy.stats
if __name__ == "__main__":
    MAP_FITS_DIR = "/home/harry/clustergnfwfit_package/data/map_fits_files"
    FNAME_BRIGHTNESS_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_map_srcfree.fits'
    FNAME_NOISE_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_ivar.fits'
    FNAME_BRIGHTNESS_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits'
    FNAME_NOISE_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_ivar.fits'
    FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

    BOLOCAM_DIR = '/home/harry/clustergnfwfit_package/data/MACS_J0025.4-1222'
    FNAME_FILTERED = 'filtered_image.fits'
    FNAME_RMS = 'filtered_image_rms.fits'
    FNAME_TRANSFER = 'filtered_image_signal_transfer_function.fits'

    FPATH_BEAM_150 = r"/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"
    FPATH_BEAM_90 = r"/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"

    # CLUSTER_NAME = 'MACSJ0025.4'

    # file paths: these fields will stay the same regardless of cluster
    fpath_dict = {
        'brightness_150': os.path.join(MAP_FITS_DIR, FNAME_BRIGHTNESS_150),
        'noise_150': os.path.join(MAP_FITS_DIR, FNAME_NOISE_150),
        'brightness_90': os.path.join(MAP_FITS_DIR, FNAME_BRIGHTNESS_90),
        'noise_90': os.path.join(MAP_FITS_DIR, FNAME_NOISE_90),
        'cmb': os.path.join(MAP_FITS_DIR, FNAME_CMB),
        'beam_150': FPATH_BEAM_150,
        'beam_90': FPATH_BEAM_90,

        'bolocam_filtered': os.path.join(BOLOCAM_DIR, FNAME_FILTERED),
        'bolocam_noise': os.path.join(BOLOCAM_DIR, FNAME_RMS),
        'bolocam_transfer': os.path.join(BOLOCAM_DIR, FNAME_TRANSFER),
    }

    # these fields will vary depending on the cluster
    dec = [-12, -22, -45]  # in degrees, minutes, seconds
    ra = [0, 25, 29.9]     # in hours, minutes, seconds
    #dec = [0, 0, 0]  # in degrees, minutes, seconds
    #ra = [0, 0, 0]     # in hours, minutes, seconds
    # ra = [0, 25, 29.9]
    map_radius = 4.8  # arcminutes
    R500 = 200  # arcseconds

    import emcee
    import numpy as np

    np.random.seed(42)
    START_OVER = True

    # must be in order
    parinfo = [
        {'parname': 'theta', 'value': 45, 'fixed': None, 'limited': [True, True], 'limits': [0., 100.]},  # theta
        {'parname': 'P0_150', 'value': -22., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_150
        {'parname': 'P0_90', 'value': -45., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_90
        {'parname': 'r_x', 'value': 200., 'fixed': None, 'limited': [True, 0], 'limits': [10., 0.]},  # r_x
        {'parname': 'r_y', 'value': 400., 'fixed': None, 'limited': [True, 0], 'limits': [10., 0.]},  # r_y
        {'parname': 'r_z', 'value': 0, 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.], 'tied': '(p[3] * p[4])**0.5'},  # r_z
        {'parname': 'x_offset', 'value': 0., 'fixed': None, 'limited': [True, True], 'limits': [-300, 300]},  # x_offset
        {'parname': 'y_offset', 'value': 0., 'fixed': None, 'limited': [True, True], 'limits': [-300, 300]},  # y_offset
        {'parname': 'c_150', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_150
        {'parname': 'c_90', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_90
        {'parname': 'P0_bolocam', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_bolocam
        {'parname': 'c_bolocam', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_bolocam
    ]

    sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_err, beam_handler_bolocam = extract_maps(fpath_dict,
                    dec, ra, map_radius, verbose=False)
    print(f"ACTPlanck Map Size: {sfl_90.shape}")
    m = mpfit_ellipsoidal_gNFW.mpfit_ellipsoidal_simultaneous(R500, beam_handler_150, beam_handler_90, sfl_150,
                            sfl_90, err_150, err_90, parinfo, bolocam_map, bolocam_err, beam_handler_bolocam, None)

    m_theta, m_p0_150, m_p0_90, m_r_x, m_r_y, m_r_z, m_x_offset, m_y_offset, m_c_150, m_c_90, m_p0_bolocam, m_c_bolocam = m.params
    print(m.params)
    # r_x is major axis, so make sure r_y < r_x
    if m_r_x < m_r_y:
        m_r_x, m_r_y = m_r_y, m_r_x
        m_theta += 90
    # mle should be theta, log_neg_p0_90, log_neg_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150
    mle = np.array([m_theta, np.cbrt(m_p0_150), np.cbrt(m_p0_90), m_r_x, m_r_y, m_x_offset, m_y_offset, m_c_90, m_c_150, np.cbrt(m_p0_bolocam), m_c_bolocam])
    print(f"Max likelihood: {mle}")

    # Initialize the walkers around max likelihood
    # nwalkers is # walkers, ndim is # parameters
    mu = np.tile(mle, (32, 1))
    coords = np.random.randn(32, len(mle)) + mu
    nwalkers, ndim = coords.shape

    theta_lower = m_theta - 90
    theta_upper = m_theta + 90
    print(f'theta bounds: {theta_lower, theta_upper}')

    # use uniform ("uninformative") priors
    def log_prior(p):
        theta, cube_root_p0_90, cube_root_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150, cube_root_p0_bolocam, c_bolocam = p
        in_bounds = theta_lower < theta < theta_upper and np.cbrt(-5000) < cube_root_p0_90 < np.cbrt(5000) and np.cbrt(-5000) < cube_root_p0_150 < np.cbrt(5000) and 10 < r_x < 1000 and 10 < r_y < r_x
        in_bounds &= -100 < offset_x < 100 and -100 < offset_y < 100 and np.cbrt(-5000) < cube_root_p0_bolocam < np.cbrt(5000)
        if in_bounds:
            # return 0
            # gaussian mean 0.8r_x and sigma 0.1r_x
            # return scipy.stats.norm.logpdf(r_y, 0.8*r_x, 0.1*r_x)
            # https://arxiv.org/pdf/1611.05192.pdf (eq 4)
            axis_ratio = r_y / r_x
            return scipy.stats.beta.logpdf(axis_ratio, 7.5, 2.31)
        return -np.inf

    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    # if we assume gaussian errors centered at the x values
    # x is sfl_90, sfl_150
    # sigma is err_90, err_150
    # x, sigmas, beam_handlers are global vars for speed
    def log_likelihood(p):
        theta, cube_root_p0_90, cube_root_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150, cube_root_p0_bolocam, c_bolocam = p
        # p0_90 = -(10**log_neg_p0_90)
        # p0_150 = -(10**log_neg_p0_150)
        p0_90 = cube_root_p0_90**3
        p0_150 = cube_root_p0_150**3
        p0_bolocam = cube_root_p0_bolocam**3
        r_z = np.sqrt(r_x*r_y)
        
        gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

        if sfl_90.shape[0] % 2 == 0:
            # can use even shape of both act and bolocam data to eval the model map only once, then rebin -> speed up
            # evaluate the bigger map
            act_map_size = (sfl_90.shape[0] + beam_handler_90.get_pad_pixels())*3
            bolocam_map_size = (bolocam_map.shape[0] + beam_handler_bolocam.get_pad_pixels())*2
            map_size = max(act_map_size, bolocam_map_size)
            model_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                                                           map_size, map_size)
            
            act_crop_amount = (map_size - act_map_size) / 2
            assert int(act_crop_amount) == act_crop_amount
            act_crop_amount = int(act_crop_amount)
            if act_crop_amount > 0:
                model_act_no_c = model_no_c[act_crop_amount:-act_crop_amount, act_crop_amount:-act_crop_amount]
            else:
                model_act_no_c = model_no_c
            model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

            model_150_no_c = model_act_no_c * p0_90
            model_90_no_c = model_act_no_c * p0_150

            model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
            model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)

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
        else:
            psf_padding_act = beam_handler_150.get_pad_pixels()
            # can use this to make the 90 model beause only P0 is different
            model_act_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                (sfl_90.shape[0] + psf_padding_act)*3, (sfl_90.shape[1] + psf_padding_act)*3)
            # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
            model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

            model_150_no_c = model_act_no_c * p0_90
            model_90_no_c = model_act_no_c * p0_150

            model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
            model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)


            # ACTUALLY, CAN EVAL AT 5 or 10 ARCSECOND RES, CHOOSE BIGGER SHAPE TO EVAL, THEN CUTOUT and REBIN
            # ACTUALLY, NOT SO SIMPLE, NEED ACT TO BE EVEN


            psf_padding_bolocam = beam_handler_bolocam.get_pad_pixels()
            # eval bolocam at 5 arcsecond res, rebin to 20
            model_bolo_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                                                            (bolocam_map.shape[0] + psf_padding_bolocam)*2, (bolocam_map.shape[1] + psf_padding_bolocam)*2)
            # evaluated at 10 arcsecond resolution, rebin to 20 arcsecond pixels
            model_bolo_no_c = ellipsoid_model.rebin_2d(model_bolo_no_c, (2, 2))

            model_bolo_no_c = model_bolo_no_c * p0_bolocam

            model_bolocam = beam_handler_bolocam.convolve2d(model_bolo_no_c + c_bolocam, cut_padding=True)

        return -0.5 * (np.sum(np.square((model_90 - sfl_90)/err_90)) + np.sum(np.square((model_150 - sfl_150)/err_150)) + np.sum(np.square((model_bolocam - bolocam_map)/bolocam_err)))
        # should be + -0.5 * np.sum(np.log(2*np.pi*np.square(sigmas))) but additive constant doesn't matter

    # The definition of the log probability function
    # x, sigmas are global vars for speed
    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
        return lp + log_likelihood(p), lp

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "emcee_backend.h5"
    backend = emcee.backends.HDFBackend(filename)
    # reset if we want to start from scratch
    if START_OVER:
        backend.reset(nwalkers, ndim)
    else:
        try:
            coords = backend.get_last_sample().coords
            print("Initial size: {0}".format(backend.iteration))
        except:
            print("Error with backend")


    # Initialize the sampler
    with Pool() as pool:
        dtype = [("log_prior", float)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend, pool=pool, blobs_dtype=dtype)


        max_n = 100000

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        old_tau = np.inf

        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(coords, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            print(f'tau: {tau}')
            print(f'Effective samples: {sampler.iteration / tau}')
            print(f'Acceptance fraction: {sampler.acceptance_fraction}')
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

