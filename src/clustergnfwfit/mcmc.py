from multiprocessing import Pool
import ellipsoid_model
import mpfit_ellipsoidal_gNFW
from extract_maps import extract_maps
import gnfw_fit_map

import os
import scipy.stats
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

# beam of width 17 pixels has smallest values which are within 1% of largest
BEAM_MAP_WIDTH = 17
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

    'filtered': os.path.join(BOLOCAM_DIR, FNAME_FILTERED),
    'noise': os.path.join(BOLOCAM_DIR, FNAME_RMS),
    'transfer': os.path.join(BOLOCAM_DIR, FNAME_TRANSFER),
}

# these fields will vary depending on the cluster
dec = [-12, -22, -45]  # in degrees, minutes, seconds
ra = [0, 25, 29.9]     # in hours, minutes, seconds
#dec = [0, 0, 0]  # in degrees, minutes, seconds
#ra = [0, 0, 0]     # in hours, minutes, seconds
# ra = [0, 25, 29.9]
map_radius = 5  # arcminutes
R500 = 200  # arcseconds

import emcee
import numpy as np

np.random.seed(42)
START_OVER = True

# must be in order
parinfo = [
    {'parname': 'theta', 'value': 45, 'fixed': None, 'limited': [True, True], 'limits': [0., 100.]},  # theta
    {'parname': 'P0_150', 'value': -22., 'fixed': None, 'limited': [0, True], 'limits': [0., -0.01]},  # P0_150
    {'parname': 'P0_90', 'value': -45., 'fixed': None, 'limited': [0, True], 'limits': [0., -0.01]},  # P0_90
    {'parname': 'r_x', 'value': 200., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # r_x
    {'parname': 'r_y', 'value': 400., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # r_y
    {'parname': 'r_z', 'value': 0, 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.], 'tied': '(p[3] * p[4])**0.5'},  # r_z
    {'parname': 'x_offset', 'value': 0., 'fixed': None, 'limited': [True, True], 'limits': [-300, 300]},  # x_offset
    {'parname': 'y_offset', 'value': 0., 'fixed': None, 'limited': [True, True], 'limits': [-300, 300]},  # y_offset
    {'parname': 'c_150', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_150
    {'parname': 'c_90', 'value': 0., 'fixed': None, 'limited': [0, 0], 'limits': [0., 0.]},  # c_90
]

sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_beam_handler = extract_maps(fpath_dict,
                dec, ra, map_radius, verbose=False)
m = mpfit_ellipsoidal_gNFW.mpfit_ellipsoidal_simultaneous(R500, beam_handler_150, beam_handler_90, sfl_150,
                        sfl_90, err_150, err_90, parinfo, [])

theta, P0_150, P0_90, r_x, r_y, r_z, x_offset, y_offset, c_150, c_90 = m.params
# r_x is major axis, so make sure r_y < r_x
if r_x < r_y:
    r_x, r_y = r_y, r_x
    theta += 90
# mle should be theta, log_neg_p0_90, log_neg_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150
mle = np.array([theta, np.cbrt(P0_150), np.cbrt(P0_90), r_x, r_y, x_offset, y_offset, c_90, c_150])
print(f"Max likelihood: {mle}")

# Initialize the walkers around max likelihood
# nwalkers is # walkers, ndim is # parameters
mu = np.tile(mle, (32, 1))
coords = np.random.randn(32, 9) + mu
nwalkers, ndim = coords.shape

theta_lower = theta - 90
theta_upper = theta + 90
print(f'theta bounds: {theta_lower, theta_upper}')

# use uniform ("uninformative") priors
def log_prior(p):
    theta, cube_root_p0_90, cube_root_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150 = p
    in_bounds = theta_lower < theta < theta_upper and np.cbrt(-5000) < cube_root_p0_90 < np.cbrt(5000) and np.cbrt(-5000) < cube_root_p0_150 < np.cbrt(5000) and 10 < r_x < 1000 and 10 < r_y < r_x
    in_bounds &= -100 < offset_x < 100 and -100 < offset_y < 100
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
    theta, cube_root_p0_90, cube_root_p0_150, r_x, r_y, offset_x, offset_y, c_90, c_150 = p
    # p0_90 = -(10**log_neg_p0_90)
    # p0_150 = -(10**log_neg_p0_150)
    p0_90 = cube_root_p0_90**3
    p0_150 = cube_root_p0_150**3
    r_z = np.sqrt(r_x*r_y)
    
    dim_pad = beam_handler_90.get_pad_pixels()
    img_height = int(sfl_90.shape[0] + dim_pad)
    # shape[1] divided by 2 because of the hstack
    img_width = int(sfl_90.shape[1] + dim_pad)
    y_90 = ellipsoid_model.eval_pixel_centers(theta, p0_90, r_x, r_y, r_z, 10, R500, offset_x, offset_y, img_height*3, img_width*3)
    y_90 = ellipsoid_model.rebin_2d(y_90, (3, 3))
    y_150 = y_90 * (p0_150/p0_90)
    y_90 += c_90
    y_150 += c_150
    y_90 = beam_handler_90.convolve2d(y_90, cut_padding=True)
    y_150 = beam_handler_150.convolve2d(y_150, cut_padding=True)

    return -0.5 * (np.sum(np.square((y_90 - sfl_90)/err_90)) + np.sum(np.square((y_150 - sfl_150)/err_150)))
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



sfl_90, sfl_150, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_beam_handler = extract_maps(fpath_dict,
                                                                                   dec, ra, map_radius, deconvolve_cmb_lmax=2000, include_bolocam=True, verbose=False)

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


import matplotlib.pyplot as plt

n = 100 * np.arange(1, index + 1)
y = autocorr[:index]
plt.plot(n, n / 100.0, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");


import corner

tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate(
    (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1
)

labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
labels += ["log prob", "log prior"]

corner.corner(all_samples, labels=labels);