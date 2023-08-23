from multiprocessing import Pool
import ellipsoid_model
import mpfit_ellipsoidal_gNFW

import os
import json
import scipy.stats


import emcee
import numpy as np

use_act = None
use_bolocam = None
use_milca = None
refit_p = None

data_90 = None
data_150 = None
data_bolocam = None
data_milca = None
beam_handler_90 = None
beam_handler_150 = None
beam_handler_bolocam = None
beam_handler_milca = None
covar_90 = None
covar_150 = None
covar_bolocam = None
covar_milca = None
covar_90_inv = None
covar_150_inv = None
covar_bolocam_inv = None
covar_milca_inv = None

lower_bounds = None
upper_bounds = None
R500 = None

# use uniform ("uninformative") priors
def log_prior(p):
    if refit_p is None:
        theta, r_x, r_y, offset_x, offset_y = p[:5]
        p = p[5:]
    if use_act:
        cube_root_p0_90, cube_root_p0_150, c_90, c_150 = p[:4]
        p = p[4:]
    if use_bolocam:
        cube_root_p0_bolocam, c_bolocam = p[:2]
        p = p[2:]
    if use_milca:
        cube_root_p0_milca = p[0]
    
    # theta, cube_root_p0_90, cube_root_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150, cube_root_p0_bolocam, c_bolocam = p
    in_bounds = True
    if refit_p is None:
        in_bounds &= lower_bounds['theta'] < theta < upper_bounds['theta'] and lower_bounds['r_x'] < r_x < upper_bounds['r_x'] and lower_bounds['r_y'] < r_y < r_x
        in_bounds &= lower_bounds['offset_x'] < offset_x < upper_bounds['offset_x'] and lower_bounds['offset_y'] < offset_y < upper_bounds['offset_y']
    if use_act:
        in_bounds &= lower_bounds['cube_root_p0_90'] < cube_root_p0_90 < upper_bounds['cube_root_p0_90'] and lower_bounds['cube_root_p0_150'] < cube_root_p0_150 < upper_bounds['cube_root_p0_150']
    if use_bolocam:
        in_bounds &= lower_bounds['cube_root_p0_bolocam'] < cube_root_p0_bolocam < upper_bounds['cube_root_p0_bolocam']
    if use_milca:
        in_bounds &= lower_bounds['cube_root_p0_milca'] < cube_root_p0_milca < upper_bounds['cube_root_p0_milca']
    if in_bounds:
        if refit_p is None:
            # https://arxiv.org/pdf/1611.05192.pdf (eq 4)
            axis_ratio = r_y / r_x
            return scipy.stats.beta.logpdf(axis_ratio, 7.5, 2.31)
        else:
            return 0
    return -np.inf

# https://emcee.readthedocs.io/en/stable/tutorials/line/
# if we assume gaussian errors centered at the x values
# x is data_90, data_150
# sigma is err_90, err_150
# x, sigmas, beam_handlers are global vars for speed
def log_likelihood(p):
    if refit_p is None:
        theta, r_x, r_y, offset_x, offset_y = p[:5]
        p = p[5:]
    else: 
        theta, r_x, r_y, r_z, offset_x, offset_y = refit_p
    if use_act:
        cube_root_p0_90, cube_root_p0_150, c_90, c_150 = p[:4]
        p0_90 = cube_root_p0_90**3
        p0_150 = cube_root_p0_150**3
        p = p[4:]
    if use_bolocam:
        cube_root_p0_bolocam, c_bolocam = p[:2]
        p0_bolocam = cube_root_p0_bolocam**3
    if use_milca:
        cube_root_p0_milca = p[0]
        p0_milca = cube_root_p0_milca**3
   
    r_z = np.sqrt(r_x*r_y)
    
    gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, R500)

    if (use_act is False or data_90.shape[0] % 2 == 0) and (use_bolocam is False or data_bolocam.shape[0] % 2 == 0) and (use_milca is False or data_milca.shape[0] % 2 == 0):
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
        diff_90 = data_90 - model_90
        chi_sq_90 = (np.reshape(diff_90, (1, -1)) @ covar_90_inv @ np.reshape(diff_90, (-1, 1))).item()
        diff_150 = data_150 - model_150
        chi_sq_150 = (np.reshape(diff_150, (1, -1)) @ covar_150_inv @ np.reshape(diff_150, (-1, 1))).item()
    else:
        chi_sq_90 = 0
        chi_sq_150 = 0
    
    if use_bolocam:
        diff_bolocam = data_bolocam - model_bolocam
        chi_sq_bolocam = (np.reshape(diff_bolocam, (1, -1)) @ covar_bolocam_inv @ np.reshape(diff_bolocam, (-1, 1))).item()
    else:
        chi_sq_bolocam = 0
    if use_milca:
        diff_milca = data_milca - model_milca
        chi_sq_milca = (np.reshape(diff_milca, (1, -1)) @ covar_milca_inv @ np.reshape(diff_milca, (-1, 1))).item()
    else:
        chi_sq_milca = 0
    # print(f"chi_sq_90: {chi_sq_90}")
    # print(f"chi_sq_150: {chi_sq_150}")
    # print(f"chi_sq_bolocam: {chi_sq_bolocam}")
    # print(f"chi_sq_milca: {chi_sq_milca}")
    # likelihood is -1/2 * chi_sq
    return -0.5 * (chi_sq_90 + chi_sq_150 + chi_sq_bolocam + chi_sq_milca)

    #return -0.5 * (np.sum(np.square((model_90 - data_90)/err_90)) + np.sum(np.square((model_150 - data_150)/err_150)) + np.sum(np.square((model_bolocam - data_bolocam)/bolocam_err)))
    # should be + -0.5 * np.sum(np.log(2*np.pi*np.square(sigmas))) but additive constant doesn't matter

# The definition of the log probability function
def log_prob(p):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    return lp + log_likelihood(p), lp

def run_mcmc(p_data_90, p_data_150, p_data_bolocam, p_data_milca, p_beam_handler_90, p_beam_handler_150, p_beam_handler_bolocam, p_beam_handler_milca, p_covar_90, p_covar_150, p_covar_bolocam, p_covar_milca, p_R500, params_dict, is_refit):
    # will run refit if first_fit_backend_dir is not None
    # refit is when we fix theta, r_x, r_y, offset_x, offset_y
    
    np.random.seed(42)
    os.environ["OMP_NUM_THREADS"] = "1"

    if is_refit:
        out_dir_path = os.path.join(params_dict['folder_path'], 'mcmc_refit')
        first_fit_backend_dir = os.path.join(params_dict['folder_path'], 'mcmc_first_fit')
    else:
        out_dir_path = os.path.join(params_dict['folder_path'], 'mcmc_first_fit')
    
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)
    with os.scandir(out_dir_path) as it:
        if any(it):
            raise Exception(f"Directory at {out_dir_path} is not empty. Please delete it or set the flag to skip fitting.")

    global use_act, use_bolocam, use_milca
    use_act = params_dict['use_act']
    use_bolocam = params_dict['use_bolocam']
    use_milca = params_dict['use_milca']
    print(f"Fitting ACT: {use_act}")
    print(f"Fitting Bolocam: {use_bolocam}")
    print(f"Fitting Milca: {use_milca}")

    p_covar_90_inv, p_covar_150_inv, p_covar_bolocam_inv, p_covar_milca_inv = None, None, None, None
    if use_act:
        p_covar_90_inv, p_covar_150_inv = np.linalg.inv(p_covar_90), np.linalg.inv(p_covar_150)
    if use_bolocam:
        p_covar_bolocam_inv = np.linalg.inv(p_covar_bolocam)
    if use_milca:
        p_covar_milca_inv = np.linalg.inv(p_covar_milca)
    local_params = (p_data_90, p_data_150, p_data_bolocam, p_data_milca, p_beam_handler_90, p_beam_handler_150, p_beam_handler_bolocam, p_beam_handler_milca, p_covar_90, p_covar_150, p_covar_bolocam, p_covar_milca, p_covar_90_inv, p_covar_150_inv, p_covar_bolocam_inv, p_covar_milca_inv, p_R500)

    # write to global, use global for speed
    global data_90, data_150, data_bolocam, data_milca, beam_handler_90, beam_handler_150, beam_handler_bolocam, beam_handler_milca, covar_90, covar_150, covar_bolocam, covar_milca, covar_90_inv, covar_150_inv, covar_bolocam_inv, covar_milca_inv, R500
    data_90, data_150, data_bolocam, data_milca, beam_handler_90, beam_handler_150, beam_handler_bolocam, beam_handler_milca, covar_90, covar_150, covar_bolocam, covar_milca, covar_90_inv, covar_150_inv, covar_bolocam_inv, covar_milca_inv, R500 = local_params
    if use_act:
        print(f"ACTPlanck Map Size: {data_90.shape}")
    if use_bolocam:
        print(f"Bolocam Map Size: {data_bolocam.shape}")
    if use_milca:
        print(f"Milca Map Size: {data_milca.shape}")


    # parameter labels for corner plot
    labels = []
    if not is_refit:
        labels += ['theta', 'r_x', 'r_y', 'offset_x', 'offset_y']
    if use_act:
        labels += ['cbrt_p0_90', 'cbrt_p0_150', 'c_90', 'c_150']
    if use_bolocam:
        labels += ['cbrt_p0_bolocam', 'c_bolocam']
    if use_milca:
        labels += ['cbrt_p0_milca']
    np.savetxt(os.path.join(out_dir_path, 'labels.txt'), labels, fmt='%s')

    if is_refit:
        reader = emcee.backends.HDFBackend(os.path.join(first_fit_backend_dir, 'backend.h5'))
        try:
            tau = reader.get_autocorr_time()
        except Exception as e:
            print(e)
            tau = reader.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        samples = reader.get_chain(discard=burnin)
        p_medians = np.median(samples, axis=(0, 1))

        m_theta, m_r_x, m_r_y, m_x_offset, m_y_offset = p_medians[:5]
        p_medians = p_medians[5:]
        if use_act:
            m_p0_90, m_p0_150, m_c_90, m_c_150 = p_medians[:4]
            p_medians = p_medians[4:]
        if use_bolocam:
            m_p0_bolocam, m_c_bolocam = p_medians[:2]
            p_medians = p_medians[2:]
        if use_milca:
            m_p0_milca = p_medians[0]
            # p_medians = p_medians[1:]

    else:
        # must be in order
        parinfo = [
            {'parname': 'theta', 'value': 45, 'fixed': None, 'limited': [True, True], 'limits': [0., 100.]},  # theta
            {'parname': 'r_x', 'value': 200., 'fixed': None, 'limited': [True, 0], 'limits': [10., 0.]},  # r_x
            {'parname': 'r_y', 'value': 400., 'fixed': None, 'limited': [True, 0], 'limits': [10., 0.]},  # r_y
            {'parname': 'r_z', 'value': 0, 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.], 'tied': '(p[1] * p[2])**0.5'},  # r_z
            {'parname': 'x_offset', 'value': 0., 'fixed': None, 'limited': [True, True], 'limits': [-300, 300]},  # x_offset
            {'parname': 'y_offset', 'value': 0., 'fixed': None, 'limited': [True, True], 'limits': [-300, 300]},  # y_offset
        ]

        if use_act:
            parinfo.extend([
                {'parname': 'P0_150', 'value': -22., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_150
                {'parname': 'P0_90', 'value': -45., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_90
                {'parname': 'c_150', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_150
                {'parname': 'c_90', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_90
            ])
        if use_bolocam:
            parinfo.extend([
                {'parname': 'P0_bolocam', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_bolocam
                {'parname': 'c_bolocam', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_bolocam
            ])
        if use_milca:
            parinfo.extend([
                {'parname': 'P0_milca', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # P0_milca
            ])

        err_150 = np.sqrt(np.diag(covar_150)) if covar_150 is not None else None
        err_90 = np.sqrt(np.diag(covar_90)) if covar_90 is not None else None
        err_bolocam = np.sqrt(np.diag(covar_bolocam)) if covar_bolocam is not None else None
        err_milca = np.sqrt(np.diag(covar_milca)) if covar_milca is not None else None
        m = mpfit_ellipsoidal_gNFW.mpfit_ellipsoidal_simultaneous(parinfo, R500, beam_handler_150, beam_handler_90, data_150,
                                data_90, err_150, err_90, data_bolocam, err_bolocam, beam_handler_bolocam, 
                                data_milca, err_milca, beam_handler_milca, use_act, use_bolocam, use_milca)

        print(m.params)
        m_p = m.params
        m_theta, m_r_x, m_r_y, m_r_z, m_x_offset, m_y_offset = m_p[:6]
        m_p = m_p[6:]
        
        if use_act:
            m_p0_150, m_p0_90, m_c_150, m_c_90 = m_p[:4]
            m_p = m_p[4:]
        if use_bolocam:
            m_p0_bolocam, m_c_bolocam = m_p[:2]
            m_p = m_p[2:]
        if use_milca:
            m_p0_milca = m_p[0]

        # r_x is major axis, so make sure r_y < r_x
        if m_r_x < m_r_y:
            m_r_x, m_r_y = m_r_y, m_r_x
            m_theta += 90
        

    global lower_bounds, upper_bounds
    lower_bounds = {}
    upper_bounds = {}
    if not is_refit:
        lower_bounds = {'theta': m_theta - 90,
                    'r_x': 10, 'r_y': 10,
                    'offset_x': -100, 'offset_y': -100, }
        upper_bounds = {'theta': m_theta + 90,
                    'r_x': 1000,
                    'offset_x': 100, 'offset_y': 100, }
        print(f"theta bounds: {lower_bounds['theta'], upper_bounds['theta']}") 
    if use_act:
        lower_bounds.update({'cube_root_p0_90': -np.cbrt(5000), 'cube_root_p0_150': -np.cbrt(5000)})
        upper_bounds.update({'cube_root_p0_90': np.cbrt(5000), 'cube_root_p0_150': np.cbrt(5000)})
    if use_bolocam:
        lower_bounds.update({'cube_root_p0_bolocam': -np.cbrt(5000)})
        upper_bounds.update({'cube_root_p0_bolocam': np.cbrt(5000)})
    if use_milca:
        lower_bounds.update({'cube_root_p0_milca': -np.cbrt(5000)})
        upper_bounds.update({'cube_root_p0_milca': np.cbrt(5000)})
    with open(os.path.join(out_dir_path, 'lower_bounds.txt'), mode='w', encoding='utf-8') as f:
        json.dump(lower_bounds, f)
    with open(os.path.join(out_dir_path, 'upper_bounds.txt'), mode='w', encoding='utf-8') as f:
        json.dump(upper_bounds, f)
    

    # mle should be [m_theta, np.cbrt(m_p0_150), np.cbrt(m_p0_90), m_r_x, m_r_y, m_x_offset, m_y_offset, m_c_90, m_c_150, np.cbrt(m_p0_bolocam), m_c_bolocam]
    mle = []
    if is_refit:
        global refit_p
        refit_p = [m_theta, m_r_x, m_r_y, np.sqrt(m_r_x*m_r_y), m_x_offset, m_y_offset]
    else:
        mle += [m_theta, m_r_x, m_r_y, m_x_offset, m_y_offset]
    if use_act:
        mle += [np.cbrt(m_p0_90), np.cbrt(m_p0_150), m_c_90, m_c_150]
    if use_bolocam:
        mle += [np.cbrt(m_p0_bolocam), m_c_bolocam]
    if use_milca:
        mle += [np.cbrt(m_p0_milca)]

    mle = np.array(mle)
    print(f"Labels: {labels}")
    print(f"Max likelihood: {mle}")

    # Initialize the walkers around max likelihood
    # nwalkers is # walkers, ndim is # parameters
    mu = np.tile(mle, (32, 1))
    coords = np.random.randn(32, len(mle)) + mu
    nwalkers, ndim = coords.shape

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
            # converged = np.all(tau * 100 < sampler.iteration)
            converged = np.all(tau * 50 < sampler.iteration)
            print(f'tau: {tau}')
            print(f'Effective samples: {sampler.iteration / tau}')
            print(f'Acceptance fraction: {sampler.acceptance_fraction}')
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
