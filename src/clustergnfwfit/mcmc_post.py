
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from datetime import datetime

from astropy import units as u
from astropy.io import fits
from astropy.cosmology import Planck15

import ellipsoid_model
import nu_and_di
import szpack_bindings

# pnames is ordered list of parameter labels, 1 for each parameter in mcmc fit
# dir_path must be dir that contains the mcmc backend and bounds JSON files
# makes folder named mcmc_trace in dir_path
def make_trace(dir_path, backend_fname, lower_bounds_fname, upper_bounds_fname):
    save_path = os.path.join(dir_path, 'mcmc_trace')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pnames = list(np.loadtxt(os.path.join(dir_path, 'labels.txt'), dtype=str))

    reader = emcee.backends.HDFBackend(os.path.join(dir_path, backend_fname))
    samples = reader.get_chain()
    log_prob_samples = reader.get_log_prob()
    log_prior_samples = reader.get_blobs()
    
    with open(os.path.join(dir_path, lower_bounds_fname), 'r', encoding='utf-8') as f_lower:
        lower_bounds = json.load(f_lower)
    with open(os.path.join(dir_path, upper_bounds_fname), 'r', encoding='utf-8') as f_upper:
        upper_bounds = json.load(f_upper)
    

    num_samples = samples.shape[0]
    nwalkers = samples.shape[1]
    ndims = samples.shape[2]
    # samples is shape (nsamples, nwalkers, ndims)

    # shape is (nsamples, nwalkers)
    chisq_samples = log_prob_samples * -2


    subplots = [list(plt.subplots(3, 1)) for _ in range(len(pnames))]
    for i, (fig, *_) in enumerate(subplots):
        pname = pnames[i]
        fig.suptitle(pname)

    # i is param #
    for i in range(ndims):
        fig, (ax1, ax2, ax3) = subplots[i]
        pdata = samples[:, :, i]
        pname = pnames[i]
        # pdata is shape (nsamples, nwalkers)
        for walker_idx in range(pdata.shape[1]):
            y = pdata[:, walker_idx]
            ax1.plot(y, alpha=0.1)
            ax2.plot(y, color='r', alpha=0.02)
            # ax2.scatter(range(num_samples), y, color='r', alpha=alphas[:, i])
        lower = lower_bounds.get(pname)
        upper = upper_bounds.get(pname)
        if lower is not None or upper is not None:
            ax1.set_ylim(bottom=lower, top=upper)
        ax2.set_ylim(ax1.get_ylim())
        ax3.plot(np.mean(pdata, axis=1))
        ax3.set_ylim(ax1.get_ylim())
        pickle_path = os.path.join(save_path, pname + '.pickle')
        with open(pickle_path, 'bw') as f:
            pickle.dump(fig, f)


    # walker chi-sqs
    fig_chisq, (ax_chisq_walkers, ax_chisq_mean) = plt.subplots(2, 1)
    fig_chisq.suptitle('chi sq')
    ax_chisq_walkers.plot(chisq_samples, alpha=0.1)
    ax_chisq_mean.plot(np.mean(chisq_samples, axis=1))
    pickle_path = os.path.join(save_path, 'chi_sq.pickle')
    with open(pickle_path, 'bw') as f:
        pickle.dump(fig_chisq, f)


    print("chain shape: {0}".format(samples.shape))
    print("log prob shape: {0}".format(log_prob_samples.shape))
    print("log prior shape: {0}".format(log_prior_samples.shape))

    return pickle_path   

# labels is list, one for each parameter
def make_corner(dir_path, backend_fname):
    save_path = os.path.join(dir_path, 'corner_plots')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    labels = list(np.loadtxt(os.path.join(dir_path, 'labels.txt'), dtype=str))
    labels += ["log prob", "log prior"]

    backend_path = os.path.join(dir_path, backend_fname)
    reader = emcee.backends.HDFBackend(backend_path)

    try:
        tau = reader.get_autocorr_time()
    except Exception as e:
        print(e)
        tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
    log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin).astype(float)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    print("flat log prior shape: {0}".format(log_prior_samples.shape))

    all_samples = np.concatenate(
        (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1
    )

    fig = corner.corner(all_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, range=[1]*len(labels))
    pickle_path = os.path.join(save_path, 'corner.pickle')
    with open(pickle_path, 'bw') as f:
        pickle.dump(fig, f)
    plt.close()

# dir_path should be the path to the directory containing the .pickle files
def show_pickled_plots(dir_path):
    for fname in os.listdir(dir_path):
        _, ext = os.path.splitext(fname)
        if ext != ".pickle":
            continue
        with open(os.path.join(dir_path, fname), 'br') as f:
            pickle.load(f)

# dir_path should be the path to the directory containing the .pickle files
def show_npy_plots(dir_path, figure_name_hook=None, pre_imshow_hook=None):
    for file_name in os.listdir(dir_path):
        _, ext = os.path.splitext(file_name)
        if ext != ".npy":
            continue
        figure_name = file_name
        if figure_name_hook is not None:
            figure_name = figure_name_hook(figure_name)
        plt.figure(figure_name)
        arr = np.load(os.path.join(dir_path, file_name))
        if pre_imshow_hook is not None:
            arr = pre_imshow_hook(arr)
        plt.imshow(arr)

def get_R2500_avg(map, arcseconds_per_pixel, R2500):
    """AKA get di value.

    Args:
        map (2d array): 
        arcseconds_per_pixel (int): length of pixel in arcseconds
        R2500 (float): cluster R2500 in arcseconds.
        Pixels within R2500 will be used in calculation.

    Returns:
        float: Returns the average of all pixels in map that are within R2500 of its center.
    
    Notes: The return value from this function can be used as a divisor
    to divide the map. This will result in a map with an average value of
    1 within R2500.

    """

    # inefficient but doesn't matter
    center_pix_x = (map.shape[1] - 1) / 2
    center_pix_y = (map.shape[0] - 1) / 2
    map = np.copy(map)
    num_in_R2500 = 0
    for pixel_y in range(map.shape[0]):
        for pixel_x in range(map.shape[1]):
            dist_x = np.abs((pixel_x - center_pix_x))
            dist_y = np.abs((pixel_y - center_pix_y))
            # convert pixel distance to arcsecond distance
            r = np.sqrt(dist_x ** 2 + dist_y ** 2) * arcseconds_per_pixel
            if r > R2500:
                map[pixel_y, pixel_x] = 0
            else:
                num_in_R2500 += 1
    return np.sum(map) / num_in_R2500

# returns best fit model normalized to di of 1, and 3 tuples (di, std of di) for 90, 150, bolocam
def get_gnfw_model_and_dis(R500, R2500, nu_90, nu_150, nu_bolocam, first_fit_backend_dir_path, second_fit_backend_dir_path):

    def get_chain(backend_path):
        reader = emcee.backends.HDFBackend(backend_path)
        try:
            tau = reader.get_autocorr_time()
        except Exception as e:
            print(e)
            tau = reader.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        samples = reader.get_chain(discard=burnin)
        return samples

    def convert_microkelvin_to_mjysr(arr, freq):
        freq = freq * u.GHz
        # from astropy.cosmology import Planck15
        equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
        return ((arr) * u.uK).to(u.MJy / u.sr, equivalencies=equiv).value

    first_fit_backend_path = os.path.join(first_fit_backend_dir_path, 'backend.h5')
    second_fit_backend_path = os.path.join(second_fit_backend_dir_path, 'backend.h5')
    # samples is shape (nsamples, nwalkers, nparameters)
    first_samples = get_chain(first_fit_backend_path)
    second_samples = get_chain(second_fit_backend_path)

    first_medians = np.median(first_samples, axis=(0, 1))
    theta, r_x, r_y, offset_x, offset_y = first_medians[:5]
    r_z = np.sqrt(r_x*r_y)

    use_act = np.genfromtxt(os.path.join(first_fit_backend_dir_path, 'use_act.txt'), dtype=bool).item()
    use_bolocam = np.genfromtxt(os.path.join(first_fit_backend_dir_path, 'use_bolocam.txt'), dtype=bool).item()

    # reshape -> (ntotalsamples, nparameters)
    second_samples = second_samples.reshape((-1, second_samples.shape[-1]))
    if use_act:
        samples_cbrt_p0_90 = second_samples[:, 0]
        samples_cbrt_p0_150 = second_samples[:, 1]
    second_samples = second_samples[:, 4:]
    if use_bolocam:
        samples_cbrt_p0_bolocam = second_samples[:, 0]

    model_no_c = ellipsoid_model.eval_pixel_centers(theta, 1, r_x, r_y, r_z, 4, R500, offset_x=offset_x, offset_y=offset_y, img_height=470, img_width=470)
    
    di_model_no_c = get_R2500_avg(model_no_c, 4, R2500)
    dis_90 = convert_microkelvin_to_mjysr(di_model_no_c * (samples_cbrt_p0_90 ** 3), nu_90)
    dis_150 = convert_microkelvin_to_mjysr(di_model_no_c * (samples_cbrt_p0_150 ** 3), nu_150)
    dis_bolocam = convert_microkelvin_to_mjysr(di_model_no_c * (samples_cbrt_p0_bolocam ** 3), nu_bolocam)

    di_90 = np.median(dis_90)
    di_150 = np.median(dis_150)
    di_bolocam = np.median(dis_bolocam)
    sigma_90 = np.std(dis_90)
    sigma_150 = np.std(dis_150)
    sigma_bolocam = np.std(dis_bolocam)

    # model_90 = model_no_c * di_90 / di_model_no_c
    # model_150 = model_no_c * di_150 / di_model_no_c
    # model_bolocam = model_no_c * di_bolocam / di_model_no_c

    model_no_c /= di_model_no_c

    return model_no_c, (di_90, sigma_90), (di_150, sigma_150), (di_bolocam, sigma_bolocam)

def make_fits(out_fits_fpath, first_fit_backend_dir_path, second_fit_backend_dir_path, param_dict):
    
    hdr = fits.Header()

    str_date, date_comment = datetime.today().strftime('%Y-%m-%d'), "Creation UTC (CCCC-MM-DD)"
    hdr['DATE'] = (str_date, date_comment)

    hdr['PRIMARY'] = 'Best-fit gNFW model'
    hdr['OBJNAME'] = param_dict['obj_name']
    hdr['EQUINOX'] = float(2000)
    hdr['PIXSIZE'] = (4, "length of pixel side (arcseconds)")
    hdr['TTYPE'] = 'R2500 normalized'
    hdr['TUNIT'] = 'keV'


    # get nus, dIs

    szpack_bindings.load_lib(param_dict['szpack_fpath'])
    T_mw, sigma_T_mw, T_pw, sigma_T_pw = nu_and_di.get_weighted_xray_temperature(param_dict)

    nu_bolocam, nu_plw, nu_pmw, nu_psw = nu_and_di.calc_bolo_and_spire_band_centers(T_pw, param_dict)
    nu_90, nu_150 = nu_and_di.get_act_band_centers(T_pw, param_dict)

    model_no_c, (di_90, sigma_90), (di_150, sigma_150), (di_bolocam, sigma_bolocam) = get_gnfw_model_and_dis(param_dict['r500'], param_dict['r2500'], nu_90, nu_150, nu_bolocam, first_fit_backend_dir_path, second_fit_backend_dir_path)
    
    tau_e = nu_and_di.fit_tau_e(nu_bolocam, nu_90, nu_150, di_bolocam, di_90, di_150, T_pw)
    print(f"tau_e: {tau_e}")
    di_spire_plw = (nu_and_di.compute_sz_spectrum(np.array([nu_plw * 1.e9]), temperature=T_pw))[0] * 100 * tau_e
    di_spire_pmw = (nu_and_di.compute_sz_spectrum(np.array([nu_pmw * 1.e9]), temperature=T_pw))[0] * 100 * tau_e
    # di_spire_psw = (nu_and_di.compute_sz_spectrum(np.array([nu_psw * 1.e9]), temperature=T_pw))[0] * 100 * tau_e

    hdr['BCAMDI'] = di_bolocam
    hdr['BCAMERR'] = sigma_bolocam
    hdr['BCAMUNIT'] = 'MJy/sr'

    hdr['ACTDI1'] = di_90
    hdr['ACTDI2'] = di_150
    hdr['ACTERR1'] = sigma_90
    hdr['ACTERR2'] = sigma_150
    hdr['ACTUNIT'] = 'MJy/sr'

    hdr['REDSHIFT'] = param_dict['redshift']
    hdr['R2500'] = param_dict['r2500'] / 60     # convert to arcmin
    hdr['R2500UNI'] = 'arcmin'

    def hms_to_deg(hours, minutes, seconds):
        return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
    def dms_to_deg(degrees, minutes, seconds):
        return degrees + minutes / 60 + seconds / (60 ** 2)

    xray_ra = hms_to_deg(*param_dict['ra'])
    xray_dec = dms_to_deg(*param_dict['dec'])

    hdr['XRAYRA0'] = xray_ra
    hdr['XRAYDEC0'] = xray_dec
    hdr['TXRAYPW'] = T_pw
    hdr['TPWERR'] = sigma_T_pw
    hdr['TXRAYMW '] = T_mw
    hdr['TMWERR'] = sigma_T_mw

    hdr['BOLONU0'] = (nu_bolocam, 'GHz')
    hdr['ACTNU01'] = (nu_90, 'GHz')
    hdr['ACTNU02'] = (nu_150, 'GHz')
    hdr['PLWNU0'] = (nu_plw, 'GHz')
    hdr['PMWNU0'] = (nu_pmw, 'GHz')


    hdr['PLWDI'] = di_spire_plw
    hdr['PMWDI'] = di_spire_pmw

    hdr['CTYPE1'] = ('RA---SFL', "Coordinate Type")
    hdr['CTYPE2'] = ('DEC---SFL', "Coordinate Type")

    hdr['CD1_1'] = (-0.00111111, "Degrees / Pixel")
    hdr['CD1_2'] = (-0.00000, "Degrees / Pixel")
    hdr['CD2_1'] = (0.00000, "Degrees / Pixel")
    hdr['CD2_2'] = (0.00111111, "Degrees / Pixel")

    hdr['CRPIX1'] = (235.500, "Reference Pixel in X")
    hdr['CRPIX2'] = (235.500, "Reference Pixel in Y")

    hdr['CRVAL1'] = (xray_ra, "R.A. (degrees) of reference pixel")
    hdr['CRVAL2'] = (xray_dec, "Declination of reference pixel")

    hdr['PV1_0'] = (0.00000000000, "Projection parameters")
    hdr['PV1_1'] = (0.00000000000, "Projection parameters")
    hdr['PV1_2'] = (0.00000000000, "Projection parameters")
    hdr['PV1_3'] = (180.000000000, "Projection parameters")
    hdr['PV1_4'] = (90.000000000, "Projection parameters")

    hdr['RADESYS'] = ('FK5', "Reference frame")

    primary = fits.PrimaryHDU(data=model_no_c, header=hdr)
    hdul = fits.HDUList([primary])
    hdul.writeto(out_fits_fpath, overwrite=True)
