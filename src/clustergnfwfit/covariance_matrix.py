import numpy as np
import scipy as sp
import scipy.stats
import scipy.ndimage
from astropy.io import fits
import os
from pixell import enmap
import warnings
import gc

from astropy.units import cds
import matplotlib.pyplot as plt

import extract_maps

def sample_covariance_matrix(samples, is_normal=False):
    # https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
    # samples is list of 1d np arrays
    samples = np.stack(samples, axis=-1)
    mean = np.mean(samples, axis=-1, keepdims=True)
    samples -= mean

    outer_sum = np.zeros((samples[..., 0].size, samples[..., 0].size), dtype=float)
    for i in range(samples.shape[-1]):
        sample = samples[..., i]
        outer = np.outer(sample, sample)
        outer_sum += outer

    normal_correction = int(not is_normal)
    return (1/(samples.shape[-1] - normal_correction)) * outer_sum

def sample_covariance_matrix_median(samples):
    # https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
    # samples is list of 1d np arrays
    samples = np.stack(samples, axis=-1)
    mean = np.mean(samples, axis=-1, keepdims=True)
    samples -= mean

    # slow to get past low memory
    n = samples[..., 0].size
    outer_median = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = np.median(samples[i] * samples[j])
            outer_median[i, j] = val
            outer_median[j, i] = val

    return outer_median

def get_covar_bolocam(fits_path, num_pairs):
    # fits_path is path to bolocam noise realizations

    hdul = fits.open(fits_path, lazy_load_hdus=False)
    realizations = [hdul[hdu].data.flatten() for hdu in range(num_pairs)]
    return np.cov(np.vstack(realizations, dtype=np.double), rowvar=False, bias=True)

    # realizations = []
    # for i in range(num_pairs):
    #     print(i)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         realization = enmap.read_fits(fits_path, hdu=i)

    #         # covar_bolocam = np.load(os.path.join(save_path, 'bolo_covar_1000.npy'))

    #         # eig_vals, eig_vecs = np.linalg.eigh(covar_bolocam)
    #         # eig_vals[eig_vals < 1] = np.inf
    #         # eig_vecs should probably be orthonormal, so can probably .T instead of .inv
    #         # covar_bolocam_reformed = eig_vecs @ np.diag(eig_vals) @ np.linalg.inv(eig_vecs)

    #         # eig_vals_inv = 1/eig_vals
    #         # covar_bolocam_inv = eig_vecs.T @ np.diag(eig_vals_inv) @ eig_vecs
    #         # np.save(os.path.join(save_path, 'bolo_covar_1000_inv_eigdecomp'), covar_bolocam_inv)

    #         # fpath = os.path.join(os.getcwd(), "run_inputs", "MACSJ0025.4.txt")
    #         # inputs = run.parse_input(fpath)
    #         # fpath = os.path.join(os.getcwd(), "run_inputs", "MACSJ0025.4.txt")
    #         # data_bolocam, beam_handler_bolocam = extract_maps.extract_bolocam_map(inputs, (5, 0, 0), (-7, 0, 0), inputs['deconvolution_map_radius'], inputs['deconvolve_cmb_lmax'])

    #         # diff = data_bolocam.ravel() - np.mean(data_bolocam)
    #         # print(diff.T @ covar_bolocam_inv @ diff)
    #         # input()

    #         # plt.figure('inv of inv covar')
    #         # plt.imshow(np.linalg.inv(covar_bolocam_inv))
    #         # plt.figure('regular covar')
    #         # plt.imshow(covar_bolocam)
    #         # plt.figure('diff')
    #         # plt.imshow(covar_bolocam - np.linalg.inv(covar_bolocam_inv))
    #         # plt.show()
            
    #         realization = realization.flatten()
    #         realization -= np.mean(realization)
    #         # print(realization.T @ covar_bolocam_inv @ realization)
    #         # input()

    #     realizations.append(realization.flatten())
    from sklearn.covariance import GraphicalLassoCV
    return GraphicalLassoCV(mode='cd', n_jobs=-2, verbose=True, enet_tol=0.15, eps=0.15).fit(np.vstack(realizations, dtype=np.double))
    return np.cov(np.vstack(realizations, dtype=np.double), rowvar=False, bias=True)
    return sample_covariance_matrix(realizations, is_normal=True)

def get_covar_act(num_pairs, batch_size, fpath_dict,
                dec, ra, pick_sample_radius, map_radius,
                deconvolve_cmb_lmax=2000, verbose=True, even_maps=True):
    
    # batches stores tuples (covariance matrix of batch (biased), number maps in batch)
    total_realizations = 0

    covar_90 = None
    covar_150 = None
    covar_realizations = 0
    
    while total_realizations < num_pairs:
        realizations_90 = []
        realizations_150 = []
        
        cmb_subtracted_90, cmb_subtracted_150 = extract_maps.extract_act_maps_covar(batch_size, fpath_dict, dec, ra, pick_sample_radius, map_radius, 30 * cds.arcsec, 0.5 * cds.deg, deconvolve_cmb_lmax)
        print(f"Current # realizations: {total_realizations}")

        for map_90, map_150 in zip(cmb_subtracted_90, cmb_subtracted_150):
            
            # import matplotlib.pyplot as plt
            # from matplotlib import cm
            blurred_90 = scipy.ndimage.gaussian_filter(map_90, (2, 2))
            blurred_150 = scipy.ndimage.gaussian_filter(map_150, (2, 2))

            iqr_90 = scipy.stats.iqr(blurred_90)
            mean_90 = np.mean(blurred_90)
            if (np.max(blurred_90) - mean_90) / iqr_90 > 3 or (mean_90 - np.min(blurred_90)) / iqr_90 > 3:
                print("Rejected")
                continue
            iqr_150 = scipy.stats.iqr(blurred_150)
            mean_150 = np.mean(blurred_150)
            if (np.max(blurred_150) - mean_150) / iqr_150 > 3 or (mean_150 - np.min(blurred_150)) / iqr_150 > 3:
                print("Rejected")
                continue

            # plt.figure('90 blurred')
            # plt.imshow(blurred_90, cmap=cm.coolwarm, vmin=-100, vmax=100)
        
            # plt.figure('150 blurred')
            # plt.imshow(blurred_150, cmap=cm.coolwarm, vmin=-100, vmax=100)
            
            # def plot_hist(data):
            #     h = 2 * sp.stats.iqr(data) * data.size**(-1/3)
            #     plt.hist(data.flatten(), bins=int((np.max(data) - np.min(data))/h), color='blue')
            #     mdata = np.ma.filled(data, np.nan)
            #     plt.axvline(x=np.nanpercentile(mdata, 25), color='blue', linestyle=':')
            #     plt.axvline(x=np.nanpercentile(mdata, 50), color='blue', linestyle=':')
            #     plt.axvline(x=np.nanpercentile(mdata, 75), color='blue', linestyle=':')
            #     mean = np.mean(data)
            #     std = np.std(data)
            #     plt.axvline(x=mean, color='red', linestyle=':')
            #     plt.axvline(x=mean - std, color='red', linestyle=':')
            #     plt.axvline(x=mean + std, color='red', linestyle=':')

            # plt.figure('hist 90 blurred')
            # plt.suptitle(f'skewness: {scipy.stats.skew(blurred_90.flatten())}')
            # plot_hist(blurred_90)

            # plt.figure('hist 150 blurred')
            # plt.suptitle(f'skewness: {scipy.stats.skew(blurred_150.flatten())}')
            # plot_hist(blurred_150)

            # plt.show()
            
            print("Accepted")
            realizations_90.append(map_90.flatten())
            realizations_150.append(map_150.flatten())
            del map_90
            del map_150

            total_realizations += 1
            if total_realizations >= num_pairs:
                break
        
        del cmb_subtracted_90
        del cmb_subtracted_150
        gc.collect()

        batch_covar_90 = np.cov(np.vstack(realizations_90, dtype=np.double), rowvar=False, bias=True)
        if covar_90 is None:
            covar_90 = batch_covar_90
        else:
            covar_90 = (covar_90 * covar_realizations + batch_covar_90 * len(realizations_90)) / total_realizations
        del batch_covar_90
        del realizations_90
        gc.collect()

        batch_covar_150 = np.cov(np.vstack(realizations_150, dtype=np.double), rowvar=False, bias=True)
        if covar_150 is None:
            covar_150 = batch_covar_150
        else:
            covar_150 = (covar_150 * covar_realizations + batch_covar_150 * len(realizations_150)) / total_realizations
        del batch_covar_150
        del realizations_150
        gc.collect()

        covar_realizations = total_realizations

        continue

    # return sample_covariance_matrix_median(realizations_90), sample_covariance_matrix_median(realizations_150)
    return covar_90, covar_150
    # return sample_covariance_matrix(realizations_90, is_normal=True), sample_covariance_matrix(realizations_150, is_normal=True)
    
    # realizations_90 = np.stack(realizations_90, axis=-1)
    # realizations_150 = np.stack(realizations_150, axis=-1)
    # mean_90 = np.mean(realizations_90, axis=-1, keepdims=True)
    # mean_150 = np.mean(realizations_150, axis=-1, keepdims=True)
    # realizations_90 -= mean_90
    # realizations_150 -= mean_150

    # outer_sum_90 = np.zeros((realizations_90[..., 0].size, realizations_90[..., 0].size), dtype=float)
    # outer_sum_150 = np.zeros((realizations_150[..., 0].size, realizations_150[..., 0].size), dtype=float)
    # for i in range(realizations_90.shape[-1]):
    #     realization_90 = realizations_90[..., i]
    #     realization_150 = realizations_150[..., i]
    #     outer_90 = np.outer(realization_90, realization_90)
    #     outer_150 = np.outer(realization_150, realization_150)

    #     outer_sum_90 += outer_90
    #     outer_sum_150 += outer_150

    # return (1/(realizations_90.shape[-1] - 1)) * outer_sum_90, (1/(realizations_150.shape[-1] - 1)) * outer_sum_150

def get_covar_milca(num_pairs, batch_size, fpath_dict,
                dec, ra, pick_sample_radius, map_radius,
                verbose=True, even_maps=True):
    
    # batches stores tuples (covariance matrix of batch (biased), number maps in batch)
    total_realizations = 0

    covar_milca = None
    covar_realizations = 0
    
    while total_realizations < num_pairs:
        realizations_milca = []
        
        sfl_milca = extract_maps.extract_milca_maps_covar(batch_size, fpath_dict, dec, ra, pick_sample_radius, map_radius, 10/3 * cds.arcmin)
        print(f"Current # realizations: {total_realizations}")

        for map_milca in sfl_milca:
            
            blurred_milca = scipy.ndimage.gaussian_filter(map_milca, (2, 2))

            iqr_milca = scipy.stats.iqr(blurred_milca)
            mean_milca = np.mean(blurred_milca)
            if (np.max(blurred_milca) - mean_milca) / iqr_milca > 3 or (mean_milca - np.min(blurred_milca)) / iqr_milca > 3:
                print("Rejected")
                continue

            print("Accepted")
            realizations_milca.append(map_milca.flatten())
            del map_milca

            total_realizations += 1
            if total_realizations >= num_pairs:
                break
        
        del sfl_milca
        gc.collect()

        batch_covar_milca = np.cov(np.vstack(realizations_milca, dtype=np.double), rowvar=False, bias=True)
        if covar_milca is None:
            covar_milca = batch_covar_milca
        else:
            covar_milca = (covar_milca * covar_realizations + batch_covar_milca * len(realizations_milca)) / total_realizations
        del batch_covar_milca
        del realizations_milca
        gc.collect()

        covar_realizations = total_realizations

        continue

    return covar_milca



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     BOLOCAM_DIR = '/home/harry/clustergnfwfit_package/data/MACS_J0025.4-1222'
#     fname_noise_real = 'filtered_image_noise_realizations.fits'
#     bolocam_fits_path = os.path.join(BOLOCAM_DIR, fname_noise_real)

#     save_path = 'data/covar_np_saves'

#     print('getting bolocam covar')
#     bolocam_covar_1000 = get_covar_bolocam(bolocam_fits_path, 1000)
    # import pickle
    # pickle.dump(bolocam_covar_1000, open(os.path.join(save_path, 'lasso_covar.pickle'), 'wb'))
    # print('done')

    # plt.figure('lasso covar')
    # plt.imshow(bolocam_covar_1000.covariance_)
    # plt.figure('lasso precision')
    # plt.imshow(bolocam_covar_1000.get_precision())

    # plt.figure('lasso, regular diff')
    # plt.imshow(bolocam_covar_1000.covariance_ - np.load(os.path.join(save_path, 'bolo_covar_1000.npy')))
    # plt.show()

#     np.save(os.path.join(save_path, 'bolo_covar_1000'), bolocam_covar_1000)

#     bolocam_covar_1000 = np.load(os.path.join(save_path, 'bolo_covar_1000.npy'))

#     print(np.linalg.inv(bolocam_covar_1000))
#     # make corr matrix later
#     yy, xx = np.mgrid[:bolocam_covar_1000.shape[0], :bolocam_covar_1000.shape[1]]
#     corr_1000 = bolocam_covar_1000[np.ix_(range(bolocam_covar_1000.shape[0]), range(bolocam_covar_1000.shape[1]))] / np.sqrt(np.diag(bolocam_covar_1000)[yy] * np.diag(bolocam_covar_1000)[xx])
    
#     diag = np.sqrt(np.diag(np.diag(bolocam_covar_1000)))
#     gaid = np.linalg.inv(diag)
#     corl = gaid @ bolocam_covar_1000 @ gaid
#     plt.figure('corr 1000')
#     plt.imshow(corl)

#     # diag_10 = np.expand_dims(np.sqrt(np.diag(bolocam_covar_10)), -1)
#     # diag_100 = np.expand_dims(np.sqrt(np.diag(bolocam_covar_100)), -1)
#     diag_1000 = np.expand_dims(np.sqrt(np.diag(bolocam_covar_1000)), -1)

#     # plt.figure('10')
#     # plt.imshow(bolocam_covar_10)
    
#     # plt.figure('diag 10')
#     # plt.imshow(diag_10, aspect='auto')

#     # plt.figure('100')
#     # plt.imshow(bolocam_covar_100)

#     # plt.figure('diag 100')
#     # plt.imshow(diag_100, aspect='auto')

#     plt.figure('1000')
#     plt.imshow(bolocam_covar_1000)

#     plt.figure('diag 1000')
#     plt.imshow(diag_1000, aspect='auto')

#     # plt.figure('diags 10 / 1000')
#     # plt.imshow(diag_10 / diag_1000, aspect='auto')
    
#     # plt.figure('diags 100 / 1000')
#     # plt.imshow(diag_100 / diag_1000, aspect='auto')

#     # print(np.median(diag_100 / diag_1000))
#     # print(np.std(diag_100 / diag_1000))


#     plt.show()

if __name__ == "__main__":
    save_path = 'data/covar_np_saves'

    act_90_covar_31000 = np.load(os.path.join(save_path, 'act_90_covar_31000_reject_3_iqr.npy'))
    act_150_covar_31000 = np.load(os.path.join(save_path, 'act_150_covar_31000_reject_3_iqr.npy'))
    eig_31000, _ = np.linalg.eig(act_90_covar_31000)
    eig_31000 = np.reshape(eig_31000, (40, 40))
    plt.figure('eig 31k')
    plt.imshow(eig_31000)
    # plt.figure('act 90 covar 31000')
    # plt.imshow(act_90_covar_31000)

    act_90_covar_30000 = np.load(os.path.join(save_path, 'act_90_covar_30000_reject_3_iqr.npy'))
    act_150_covar_30000 = np.load(os.path.join(save_path, 'act_150_covar_30000_reject_3_iqr.npy'))
    eig_30000, _ = np.linalg.eig(act_90_covar_30000)
    eig_30000 = np.reshape(eig_30000, (40, 40))
    plt.figure('eig 30k')
    plt.imshow(eig_30000)
    # plt.figure('act 90 covar 30000')
    # plt.imshow(act_90_covar_30000)
    
    act_90_covar_20000 = np.load(os.path.join(save_path, 'act_90_covar_20000_reject_3_iqr.npy'))
    act_150_covar_20000 = np.load(os.path.join(save_path, 'act_150_covar_20000_reject_3_iqr.npy'))
    eig_20000, _ = np.linalg.eig(act_90_covar_20000)
    eig_20000 = np.reshape(eig_20000, (40, 40))
    plt.figure('eig 20k')
    plt.imshow(eig_20000)
    # plt.figure('act 90 covar 20000')
    # plt.imshow(act_90_covar_20000)

    act_90_covar_19000 = np.load(os.path.join(save_path, 'act_90_covar_19000_reject_3_iqr.npy'))
    act_150_covar_19000 = np.load(os.path.join(save_path, 'act_150_covar_19000_reject_3_iqr.npy'))
    eig_19000, _ = np.linalg.eig(act_90_covar_19000)
    eig_19000 = np.reshape(eig_19000, (40, 40))
    plt.figure('eig 19k')
    plt.imshow(eig_19000)
    # plt.figure('act 90 covar 19000')
    # plt.imshow(act_90_covar_19000)

    act_90_covar_15000 = np.load(os.path.join(save_path, 'act_90_covar_15000_reject_3_iqr.npy'))
    act_150_covar_15000 = np.load(os.path.join(save_path, 'act_150_covar_15000_reject_3_iqr.npy'))
    eig_15000, _ = np.linalg.eig(act_90_covar_15000)
    eig_15000 = np.reshape(eig_15000, (40, 40))
    plt.figure('eig 15k')
    plt.imshow(eig_15000)  

    act_90_covar_5000 = np.load(os.path.join(save_path, 'act_90_covar_5000_reject_3_iqr.npy'))
    act_150_covar_5000 = np.load(os.path.join(save_path, 'act_150_covar_5000_reject_3_iqr.npy'))
    eig_5000, _ = np.linalg.eig(act_90_covar_5000)
    eig_5000 = np.reshape(eig_5000, (40, 40))
    plt.figure('eig 5k')
    plt.imshow(eig_5000)    
    
    act_90_covar_10000 = np.load(os.path.join(save_path, 'act_90_covar_10000_reject_3_iqr.npy'))
    act_150_covar_10000 = np.load(os.path.join(save_path, 'act_150_covar_10000_reject_3_iqr.npy'))
    eig_10000, _ = np.linalg.eig(act_90_covar_10000)
    eig_10000 = np.reshape(eig_10000, (40, 40))
    plt.figure('eig 10k')
    plt.imshow(eig_10000)  

    act_90_covar_2000 = np.load(os.path.join(save_path, 'act_90_covar_2000_reject_3_iqr.npy'))
    act_150_covar_2000 = np.load(os.path.join(save_path, 'act_150_covar_2000_reject_3_iqr.npy'))
    eig_2000, _ = np.linalg.eig(act_90_covar_2000)
    eig_2000 = np.reshape(eig_2000, (40, 40))
    plt.figure('eig 2k')
    plt.imshow(eig_2000)

    plt.figure('eig 31k / 2k')
    plt.imshow(eig_31000 / eig_2000)

    plt.figure('eig 31k / 30k')
    plt.imshow(eig_31000 / eig_30000)

    fig, ax = plt.subplots()
    ax.plot(eig_31000.ravel(), label='31k')
    ax.plot(eig_30000.ravel(), label='30k')
    ax.plot(eig_20000.ravel(), label='20k')
    ax.plot(eig_19000.ravel(), label='19k')
    ax.plot(eig_15000.ravel(), label='15k')
    ax.plot(eig_10000.ravel(), label='10k')
    ax.plot(eig_5000.ravel(), label='5k')
    ax.plot(eig_2000.ravel(), label='2k')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
    pick_sample_radius = 10 * cds.deg
    map_radius = 10 * cds.arcmin # arcminutes
    R500 = 200  # arcseconds

    save_path = 'data/covar_np_saves'

    # DELETE LATER
    act_90_covar, act_150_covar = get_covar_act(15000, 3333, fpath_dict,
                                    dec, ra, pick_sample_radius, map_radius,
                                    deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    np.save(os.path.join(save_path, 'act_90_covar_15000_reject_3_iqr'), act_90_covar)
    np.save(os.path.join(save_path, 'act_150_covar_15000_reject_3_iqr'), act_150_covar)
    plt.figure('act 90 covar')
    plt.imshow(act_90_covar)
    plt.figure('act 150 covar')
    plt.imshow(act_150_covar)
    plt.show()

    # batch_size = 1000
    # act_90_covar_10, act_150_covar_10 = get_covar_act(10, batch_size, fpath_dict,
    #                                 dec, ra, pick_sample_radius, map_radius,
    #                                 deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    # np.save(os.path.join(save_path, 'act_90_covar_10'), act_90_covar_10)
    # np.save(os.path.join(save_path, 'act_150_covar_10'), act_150_covar_10)

    # act_90_covar_100, act_150_covar_100 = get_covar_act(100, batch_size, fpath_dict,
    #                                 dec, ra, pick_sample_radius, map_radius,
    #                                 deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    # np.save(os.path.join(save_path, 'act_90_covar_100'), act_90_covar_100)
    # np.save(os.path.join(save_path, 'act_150_covar_100'), act_150_covar_100)

    # act_90_covar_1000, act_150_covar_1000 = get_covar_act(1000, batch_size, fpath_dict,
    #                                 dec, ra, pick_sample_radius, map_radius,
    #                                 deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    # np.save(os.path.join(save_path, 'act_90_covar_1000'), act_90_covar_1000)
    # np.save(os.path.join(save_path, 'act_150_covar_1000'), act_150_covar_1000)


    act_90_covar_10 = np.load(os.path.join(save_path, 'act_90_covar_10.npy'))
    act_150_covar_10 = np.load(os.path.join(save_path, 'act_150_covar_10.npy'))
    act_90_covar_100 = np.load(os.path.join(save_path, 'act_90_covar_100.npy'))
    act_150_covar_100 = np.load(os.path.join(save_path, 'act_150_covar_100.npy'))
    act_90_covar_1000 = np.load(os.path.join(save_path, 'act_90_covar_1000.npy'))
    act_150_covar_1000 = np.load(os.path.join(save_path, 'act_150_covar_1000.npy'))
    act_90_covar_2000 = np.load(os.path.join(save_path, 'act_90_covar_2000.npy'))
    act_150_covar_2000 = np.load(os.path.join(save_path, 'act_150_covar_2000.npy'))

    # make corr matrix later

    diag_act_90_covar_10 = np.expand_dims(np.sqrt(np.diag(act_90_covar_10)), -1)
    diag_act_150_covar_10 = np.expand_dims(np.sqrt(np.diag(act_150_covar_10)), -1)
    diag_act_90_covar_100 = np.expand_dims(np.sqrt(np.diag(act_90_covar_100)), -1)
    diag_act_150_covar_100 = np.expand_dims(np.sqrt(np.diag(act_150_covar_100)), -1)
    diag_act_90_covar_1000 = np.expand_dims(np.sqrt(np.diag(act_90_covar_1000)), -1)
    diag_act_150_covar_1000 = np.expand_dims(np.sqrt(np.diag(act_150_covar_1000)), -1)


    plt.figure('90 10')
    plt.imshow(act_90_covar_10)
    
    plt.figure('90 diag 10')
    plt.imshow(diag_act_90_covar_10, aspect='auto')

    plt.figure('150 10')
    plt.imshow(act_90_covar_10)
    
    plt.figure('150 diag 10')
    plt.imshow(diag_act_150_covar_10, aspect='auto')

    plt.figure('90 100')
    plt.imshow(act_90_covar_100)

    plt.figure('90 diag 100')
    plt.imshow(diag_act_90_covar_100, aspect='auto')

    plt.figure('150 100')
    plt.imshow(act_150_covar_100)

    plt.figure('150 diag 100')
    plt.imshow(diag_act_150_covar_100, aspect='auto')

    plt.figure('90 1000')
    plt.imshow(act_90_covar_1000)

    plt.figure('90 diag 1000')
    plt.imshow(diag_act_90_covar_1000, aspect='auto')

    plt.figure('150 1000')
    plt.imshow(act_150_covar_1000)

    plt.figure('150 diag 1000')
    plt.imshow(diag_act_150_covar_1000, aspect='auto')

    plt.figure('90 diags 10 / 1000')
    plt.imshow(diag_act_90_covar_10 / diag_act_90_covar_1000, aspect='auto')
    
    plt.figure('diags 100 / 1000')
    plt.imshow(diag_act_90_covar_100 / diag_act_90_covar_1000, aspect='auto')


    plt.show()
