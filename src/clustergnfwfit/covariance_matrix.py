import numpy as np
import scipy as sp
import scipy.stats
import scipy.ndimage
from astropy.io import fits
import os
from pixell import enmap
import warnings

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

def get_covar_bolocam_wiki(fits_path, num_pairs):
    realizations = []
    for i in range(num_pairs):
        print(i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            realization = enmap.read_fits(fits_path, hdu=i)
        realizations.append(realization.flatten())
    return sample_covariance_matrix(realizations, is_normal=False)

def get_covar_act(num_pairs, batch_size, fpath_dict,
                dec, ra, pick_sample_radius, map_radius,
                deconvolve_cmb_lmax=2000, verbose=True, even_maps=True):
    
    realizations_90 = []
    realizations_150 = []
    while len(realizations_90) < num_pairs:
        cmb_subtracted_90, cmb_subtracted_150 = extract_maps.extract_act_maps_for_covar(batch_size, fpath_dict, dec, ra, pick_sample_radius, map_radius, deconvolve_cmb_lmax, verbose, even_maps)
        print(len(realizations_90))
        
        for map_90, map_150 in zip(cmb_subtracted_90, cmb_subtracted_150):
            blurred_90 = scipy.ndimage.gaussian_filter(map_90, (1, 1))
            blurred_150 = scipy.ndimage.gaussian_filter(map_150, (1, 1))
            print(np.mean(map_90))
            print(np.mean(map_150))

            z, p = scipy.stats.skewtest(blurred_90, axis=None)
            print(f"skew 90 blurred; z: {z}, p: {p}")
            z, p = scipy.stats.kurtosistest(blurred_90, axis=None)
            print(f"kurtosis 90 blurred; z: {z}, p: {p}")
            z, p = scipy.stats.skewtest(blurred_150, axis=None)
            print(f"skew 150 blurred; z: {z}, p: {p}")
            z, p = scipy.stats.kurtosistest(blurred_150, axis=None)
            print(f"kurtosis 150 blurred; z: {z}, p: {p}")

            # z, p = scipy.stats.skewtest(map_90, axis=None)
            # print(f"skew 90; z: {z}, p: {p}")
            # z, p = scipy.stats.kurtosistest(map_90, axis=None)
            # print(f"kurtosis 90; z: {z}, p: {p}")
            # z, p = scipy.stats.skewtest(map_150, axis=None)
            # print(f"skew 150; z: {z}, p: {p}")
            # z, p = scipy.stats.kurtosistest(map_150, axis=None)
            # print(f"kurtosis 150; z: {z}, p: {p}")
            from matplotlib import cm
            plt.figure('90 blurred')
            plt.imshow(blurred_90, cmap=cm.coolwarm, vmin=-100, vmax=100)

            plt.figure('150 blurred')
            plt.imshow(blurred_150, cmap=cm.coolwarm, vmin=-100, vmax=100)


            plt.show()


            
            print("Accepted")
            realizations_90.append(cmb_subtracted_90.flatten())
            realizations_150.append(cmb_subtracted_150.flatten())
            continue
        continue


        # blurred_90 -= np.median(blurred_90)
        # blurred_150 -= np.median(blurred_150)

        # norm_max = 100
        # norm_90 = (np.abs(blurred_90))/(norm_max)
        # norm_150 = (np.abs(blurred_150))/(norm_max)
        # gray_90 = np.sqrt(np.clip(norm_90, 0, 1) * 255)
        # gray_150 = np.sqrt(np.clip(norm_150, 0, 1) * 255)

        from matplotlib import cm

        # plt.figure('gray 90')
        # plt.imshow(gray_90, vmin=0, vmax=255, cmap=cm.gray)

        # plt.figure('gray 150')
        # plt.imshow(gray_150, vmin=0, vmax=255, cmap=cm.gray)


        # def mask_blobs(img, fig_name):
        #     def get_circle_mask(cx, cy, r, n):
        #         y,x = np.ogrid[-cy:n-cy, -cx:n-cx]
        #         mask = x*x + y*y <= r*r
        #         return mask

        #     from skimage.feature import blob_dog
        #     plt.figure(fig_name)
        #     plt.imshow(img, cmap=cm.gray, vmin=0, vmax=255)
        #     masks = []

        #     blobs_dog = blob_dog(img, overlap=1, max_sigma=10)
        #     for blob in blobs_dog:
        #         y, x, r = blob
        #         c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
        #         plt.gca().add_patch(c)
        #         masks.append(get_circle_mask(x, y, r, img.shape[0]))
        #     mask = np.bitwise_or.reduce(np.array(masks))
        #     return mask

        plt.figure('90 blurred')
        plt.imshow(blurred_90, cmap=cm.coolwarm, vmin=-100, vmax=100)

        # mask_90 = mask_blobs(gray_90, '90 circled')
        # plt.figure('90 mask')
        # plt.imshow(mask_90, cmap=cm.gray)
                

        
    
        plt.figure('150 blurred')
        plt.imshow(blurred_150, cmap=cm.coolwarm, vmin=-100, vmax=100)
        
        # mask_150 = mask_blobs(gray_150, '150 circled')
        # plt.figure('150 mask')
        # plt.imshow(mask_150, cmap=cm.gray)
        
        def plot_hist(data):
            h = 2 * sp.stats.iqr(data) * data.size**(-1/3)
            plt.hist(data.flatten(), bins=int((np.max(data) - np.min(data))/h), color='blue')
            mdata = np.ma.filled(data, np.nan)
            plt.axvline(x=np.nanpercentile(mdata, 25), color='blue', linestyle=':')
            plt.axvline(x=np.nanpercentile(mdata, 50), color='blue', linestyle=':')
            plt.axvline(x=np.nanpercentile(mdata, 75), color='blue', linestyle=':')
            mean = np.mean(data)
            std = np.std(data)
            plt.axvline(x=mean, color='red', linestyle=':')
            plt.axvline(x=mean - std, color='red', linestyle=':')
            plt.axvline(x=mean + std, color='red', linestyle=':')

        plt.figure('hist 90 blurred')
        plt.suptitle(f'skewness: {scipy.stats.skew(blurred_90.flatten())}')
        plot_hist(blurred_90)

        plt.figure('hist 150 blurred')
        plt.suptitle(f'skewness: {scipy.stats.skew(blurred_150.flatten())}')
        plot_hist(blurred_150)

        # blurred_90_masked = np.ma.masked_array(blurred_90, mask=mask_90)
        # plt.figure('hist 90 blurred masked')
        # plt.suptitle(f'skewness: {scipy.stats.skew(blurred_90_masked.flatten())}')
        # plot_hist(blurred_90_masked)

        # blurred_150_masked = np.ma.masked_array(blurred_150, mask=mask_150)
        # plt.figure('hist 150 blurred masked')
        # plt.suptitle(f'skewness: {scipy.stats.skew(blurred_150_masked.flatten())}')
        # plot_hist(blurred_150_masked)

        plt.show()
    
    return sample_covariance_matrix(realizations_90, is_normal=False), sample_covariance_matrix(realizations_150, is_normal=False)
    
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




# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     BOLOCAM_DIR = '/home/harry/clustergnfwfit_package/data/MACS_J0025.4-1222'
#     fname_noise_real = 'filtered_image_noise_realizations.fits'
#     bolocam_fits_path = os.path.join(BOLOCAM_DIR, fname_noise_real)

#     save_path = 'data/covar_np_saves'

#     # bolocam_covar_10 = get_covar_bolocam_wiki(bolocam_fits_path, 10)
#     # np.save(os.path.join(save_path, 'bolo_covar_10_0'), bolocam_covar_10)
#     # bolocam_covar_100 = get_covar_bolocam_wiki(bolocam_fits_path, 100)
#     # np.save(os.path.join(save_path, 'bolo_covar_100_0'), bolocam_covar_100)
#     # bolocam_covar_1000 = get_covar_bolocam_wiki(bolocam_fits_path, 1000)
#     # np.save(os.path.join(save_path, 'bolo_covar_1000_0'), bolocam_covar_1000)

#     bolocam_covar_10 = np.load(os.path.join(save_path, 'bolo_covar_10_0.npy'))
#     bolocam_covar_100 = np.load(os.path.join(save_path, 'bolo_covar_100_0.npy'))
#     bolocam_covar_1000 = np.load(os.path.join(save_path, 'bolo_covar_1000_0.npy'))

#     # print(np.linalg.inv(bolocam_covar_1000))
#     # make corr matrix later
#     # yy, xx = np.mgrid[:bolocam_covar_1000.shape[0], :bolocam_covar_1000.shape[1]]
#     # corr_1000 = bolocam_covar_1000[np.ix_(range(bolocam_covar_1000.shape[0]), range(bolocam_covar_1000.shape[1]))] / np.sqrt(np.diag(bolocam_covar_1000)[yy] * np.diag(bolocam_covar_1000)[xx])
    
#     diag = np.sqrt(np.diag(np.diag(bolocam_covar_1000)))
#     gaid = np.linalg.inv(diag)
#     corl = gaid @ bolocam_covar_1000 @ gaid
#     plt.figure('corr 1000')
#     plt.imshow(corl)

#     diag_10 = np.expand_dims(np.sqrt(np.diag(bolocam_covar_10)), -1)
#     diag_100 = np.expand_dims(np.sqrt(np.diag(bolocam_covar_100)), -1)
#     diag_1000 = np.expand_dims(np.sqrt(np.diag(bolocam_covar_1000)), -1)

#     plt.figure('10')
#     plt.imshow(bolocam_covar_10)
    
#     plt.figure('diag 10')
#     plt.imshow(diag_10, aspect='auto')

#     plt.figure('100')
#     plt.imshow(bolocam_covar_100)

#     plt.figure('diag 100')
#     plt.imshow(diag_100, aspect='auto')

#     plt.figure('1000')
#     plt.imshow(bolocam_covar_1000)

#     plt.figure('diag 1000')
#     plt.imshow(diag_1000, aspect='auto')

#     plt.figure('diags 10 / 1000')
#     plt.imshow(diag_10 / diag_1000, aspect='auto')
    
#     plt.figure('diags 100 / 1000')
#     plt.imshow(diag_100 / diag_1000, aspect='auto')

#     print(np.median(diag_100 / diag_1000))
#     print(np.std(diag_100 / diag_1000))


#     plt.show()

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

    # act_90_covar_10, act_150_covar_10 = get_covar_act(1000, fpath_dict,
    #                                 dec, ra, map_radius,
    #                                 deconvolve_cmb_lmax=2000, verbose=False, even_maps=True)

    batch_size = 100
    act_90_covar_10, act_150_covar_10 = get_covar_act(10, batch_size, fpath_dict,
                                    dec, ra, pick_sample_radius, map_radius,
                                    deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    np.save(os.path.join(save_path, 'act_90_covar_10'), act_90_covar_10)
    np.save(os.path.join(save_path, 'act_150_covar_10'), act_150_covar_10)

    act_90_covar_100, act_150_covar_100 = get_covar_act(100, batch_size, fpath_dict,
                                    dec, ra, pick_sample_radius, map_radius,
                                    deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    np.save(os.path.join(save_path, 'act_90_covar_100'), act_90_covar_100)
    np.save(os.path.join(save_path, 'act_150_covar_100'), act_150_covar_100)

    act_90_covar_1000, act_150_covar_1000 = get_covar_act(1000, batch_size, fpath_dict,
                                    dec, ra, pick_sample_radius, map_radius,
                                    deconvolve_cmb_lmax=2000, verbose=True, even_maps=True)
    np.save(os.path.join(save_path, 'act_90_covar_1000'), act_90_covar_1000)
    np.save(os.path.join(save_path, 'act_150_covar_1000'), act_150_covar_1000)


    act_90_covar_10 = np.load(os.path.join(save_path, 'act_90_covar_10.npy'))
    act_150_covar_10 = np.load(os.path.join(save_path, 'act_150_covar_10.npy'))
    act_90_covar_100 = np.load(os.path.join(save_path, 'act_90_covar_100.npy'))
    act_150_covar_100 = np.load(os.path.join(save_path, 'act_150_covar_100.npy'))
    act_90_covar_1000 = np.load(os.path.join(save_path, 'act_90_covar_1000.npy'))
    act_150_covar_1000 = np.load(os.path.join(save_path, 'act_150_covar_1000.npy'))


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
