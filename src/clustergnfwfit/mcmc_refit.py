from multiprocessing import Pool
import ellipsoid_model
import mpfit_ellipsoidal_gNFW

import os
import json
import scipy.stats


import emcee
import numpy as np

data_90 = None
data_150 = None
data_bolocam = None
beam_handler_90 = None
beam_handler_150 = None
beam_handler_bolocam = None
covar_90 = None
covar_150 = None
covar_bolocam = None
covar_90_inv = None
covar_150_inv = None
covar_bolocam_inv = None

theta, r_x, r_y, r_z, offset_x, offset_y = None, None, None, None, None, None
lower_bounds = None
upper_bounds = None
R500 = None

# use uniform ("uninformative") priors
def log_prior(p):
    cube_root_p0_90, cube_root_p0_150, cube_root_p0_bolocam, c_90, c_150, c_bolocam = p
    in_bounds = lower_bounds['theta'] < theta < upper_bounds['theta'] and lower_bounds['cube_root_p0_90'] < cube_root_p0_90 < upper_bounds['cube_root_p0_90'] and lower_bounds['cube_root_p0_150'] < cube_root_p0_150 < upper_bounds['cube_root_p0_150']
    in_bounds &= lower_bounds['cube_root_p0_bolocam'] < cube_root_p0_bolocam < upper_bounds['cube_root_p0_bolocam']
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
# x is data_90, data_150
# sigma is err_90, err_150
# x, sigmas, beam_handlers are global vars for speed
def log_likelihood(p):
    cube_root_p0_90, cube_root_p0_150, cube_root_p0_bolocam, c_90, c_150, c_bolocam = p
    # p0_90 = -(10**log_neg_p0_90)
    # p0_150 = -(10**log_neg_p0_150)
    p0_90 = cube_root_p0_90**3
    p0_150 = cube_root_p0_150**3
    p0_bolocam = cube_root_p0_bolocam**3
    r_z = np.sqrt(r_x*r_y)
    
    gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

    if data_90.shape[0] % 2 == 0:
        # can use even shape of both act and bolocam data to eval the model map only once, then rebin -> speed up
        # evaluate the bigger map
        act_map_size = (data_90.shape[0] + beam_handler_90.get_pad_pixels())*3
        bolocam_map_size = (data_bolocam.shape[0] + beam_handler_bolocam.get_pad_pixels())*2
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

        model_90_no_c = model_act_no_c * p0_90
        model_150_no_c = model_act_no_c * p0_150

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
                            (data_90.shape[0] + psf_padding_act)*3, (data_90.shape[1] + psf_padding_act)*3)
        # evaluated at 10 arcsecond resolution, rebin to 30 arcsecond pixels
        model_act_no_c = ellipsoid_model.rebin_2d(model_act_no_c, (3, 3))

        model_90_no_c = model_act_no_c * p0_90
        model_150_no_c = model_act_no_c * p0_150

        model_150 = beam_handler_150.convolve2d(model_150_no_c + c_150, cut_padding=True)
        model_90 = beam_handler_90.convolve2d(model_90_no_c + c_90, cut_padding=True)


        psf_padding_bolocam = beam_handler_bolocam.get_pad_pixels()
        # eval bolocam at 5 arcsecond res, rebin to 20
        model_bolo_no_c = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, 10, offset_x, offset_y,
                                                                        (data_bolocam.shape[0] + psf_padding_bolocam)*2, (data_bolocam.shape[1] + psf_padding_bolocam)*2)
        # evaluated at 10 arcsecond resolution, rebin to 20 arcsecond pixels
        model_bolo_no_c = ellipsoid_model.rebin_2d(model_bolo_no_c, (2, 2))

        model_bolo_no_c = model_bolo_no_c * p0_bolocam

        model_bolocam = beam_handler_bolocam.convolve2d(model_bolo_no_c + c_bolocam, cut_padding=True)

    diff_90 = data_90 - model_90
    diff_150 = data_150 - model_150
    diff_bolocam = data_bolocam - model_bolocam
    chi_sq_90 = (np.reshape(diff_90, (1, -1)) @ covar_90_inv @ np.reshape(diff_90, (-1, 1))).item()
    chi_sq_150 = (np.reshape(diff_150, (1, -1)) @ covar_150_inv @ np.reshape(diff_150, (-1, 1))).item()
    chi_sq_bolocam = (np.reshape(diff_bolocam, (1, -1)) @ covar_bolocam_inv @ np.reshape(diff_bolocam, (-1, 1))).item()
    # print(f"chi_sq_90: {chi_sq_90}")
    # print(f"chi_sq_150: {chi_sq_150}")
    # print(f"chi_sq_bolocam: {chi_sq_bolocam}")
    # likelihood is -1/2 * chi_sq
    return -0.5 * (chi_sq_90 + chi_sq_150 + chi_sq_bolocam)

    #return -0.5 * (np.sum(np.square((model_90 - data_90)/err_90)) + np.sum(np.square((model_150 - data_150)/err_150)) + np.sum(np.square((model_bolocam - data_bolocam)/bolocam_err)))
    # should be + -0.5 * np.sum(np.log(2*np.pi*np.square(sigmas))) but additive constant doesn't matter

# The definition of the log probability function
def log_prob(p):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    return lp + log_likelihood(p), lp

def run_mcmc_refit(p_data_90, p_data_150, p_data_bolocam, p_beam_handler_90, p_beam_handler_150, p_beam_handler_bolocam, p_covar_90, p_covar_150, p_covar_bolocam, p_R500, first_fit_backend_dir, out_dir_path):
    np.random.seed(42)
    os.environ["OMP_NUM_THREADS"] = "1"

    os.mkdir(out_dir_path)

    reader = emcee.backends.HDFBackend(os.path.join(first_fit_backend_dir, 'backend.h5'))
    try:
        tau = reader.get_autocorr_time()
    except Exception as e:
        print(e)
        tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin)
    p_medians = np.median(samples, axis=(0, 1))

    # Initialize the walkers around first fit's parameters
    # nwalkers is # walkers, ndim is # parameters
    global theta, r_x, r_y, offset_x, offset_y, r_z
    theta, start_cbrt_p0_90, start_cbrt_p0_150, r_x, r_y, offset_x, offset_y, start_c_90, start_c_150, start_cbrt_p0_bolocam, start_c_bolocam = p_medians
    r_z = np.sqrt(r_x*r_y)
    
    start_params = np.array([start_cbrt_p0_90, start_cbrt_p0_150, start_cbrt_p0_bolocam, start_c_90, start_c_150, start_c_bolocam])
    mu = np.tile(start_params, (32, 1))
    coords = np.random.randn(*mu.shape) + mu
    nwalkers, ndim = coords.shape


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

    local_params = (p_data_90, p_data_150, p_data_bolocam, p_beam_handler_90, p_beam_handler_150, p_beam_handler_bolocam, p_covar_90, p_covar_150, p_covar_bolocam, np.linalg.inv(p_covar_90), np.linalg.inv(p_covar_150), np.linalg.inv(p_covar_bolocam), p_R500)

    # write to global, use global
    global data_90, data_150, data_bolocam, beam_handler_90, beam_handler_150, beam_handler_bolocam, covar_90, covar_150, covar_bolocam, covar_90_inv, covar_150_inv, covar_bolocam_inv, R500
    data_90, data_150, data_bolocam, beam_handler_90, beam_handler_150, beam_handler_bolocam, covar_90, covar_150, covar_bolocam, covar_90_inv, covar_150_inv, covar_bolocam_inv, R500 = local_params


    global lower_bounds, upper_bounds
    with open(os.path.join(first_fit_backend_dir, 'lower_bounds.txt'), mode='r', encoding='utf-8') as f:
        lower_bounds = json.load(f)
    with open(os.path.join(first_fit_backend_dir, 'upper_bounds.txt'), mode='r', encoding='utf-8') as f:
        upper_bounds = json.load(f)
    
    # make new bounds
    lower_bounds['theta'] = theta - 90
    upper_bounds['theta'] = theta + 90
    with open(os.path.join(out_dir_path, 'lower_bounds.txt'), mode='w', encoding='utf-8') as f:
        json.dump(lower_bounds, f)
    with open(os.path.join(out_dir_path, 'upper_bounds.txt'), mode='w', encoding='utf-8') as f:
        json.dump(upper_bounds, f)
    
    print(f"theta bounds: {lower_bounds['theta'], upper_bounds['theta']}")
    


    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend_out_fpath = os.path.join(out_dir_path, 'backend.h5')
    backend = emcee.backends.HDFBackend(backend_out_fpath)
    # reset if we want to start from scratch
    backend.reset(nwalkers, ndim)
    # else:
    #     try:
    #         coords = backend.get_last_sample().coords
    #         print("Initial size: {0}".format(backend.iteration))
    #     except:
    #         print("Error with backend")


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

