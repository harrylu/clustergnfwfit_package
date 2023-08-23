import scipy.constants
import numpy as np
import healpy as hp
from astropy.io import fits
from pixell import enmap, reproject, utils
import matplotlib.pyplot as plt
import time
import os

import ellipsoid_model
import beam_utils
import run
import extract_maps
import covariance_matrix

from astropy.units import cds
cds.enable()

# if __name__ == "__main__":
#     np_save_dir = '/home/harry/clustergnfwfit_package/data/covar_np_saves'
#     covar_milca_100k = np.load(os.path.join(np_save_dir, 'covar_milca_100k.npy'))
#     covar_milca_20k = np.load(os.path.join(np_save_dir, 'covar_milca_20k.npy'))
#     plt.figure('100k')
#     plt.imshow(np.diag(np.diag(covar_milca_100k)))
#     plt.figure('20k')
#     plt.imshow(np.diag(np.diag(covar_milca_20k)))
#     plt.figure('100k / 20k')
#     plt.imshow(np.diag(np.diag(covar_milca_100k)) / np.diag(np.diag(covar_milca_20k)))
#     print(np.diag(covar_milca_100k) / np.diag(covar_milca_20k))
#     plt.show()



if __name__ == "__main__":
    # fpath = '/home/harry/clustergnfwfit_package/run_inputs/RXJ1347.5.txt'
    fpath = '/home/harry/clustergnfwfit_package/run_inputs/MACSJ0025.4.txt'
    inputs = run.parse_input(fpath)
    # print(extract_maps.hms_to_deg(*inputs['ra']))
    # print(extract_maps.dms_to_deg(*inputs['dec']))

    # np_save_dir = '/home/harry/clustergnfwfit_package/data/covar_np_saves'
    # inputs['covar_num_samples'] = int(100e3)
    # inputs['covar_batch_size'] = int(21e3)

    # covar_milca = covariance_matrix.get_covar_milca(inputs['covar_num_samples'], inputs['covar_batch_size'], inputs,
    #                                                             inputs['dec'], inputs['ra'], inputs['covar_pick_sample_radius'], inputs['map_radius'],
    #                                                             verbose=True, even_maps=True)
    # np.save(os.path.join(np_save_dir, 'covar_milca_100k.npy'), covar_milca)
    # plt.figure('covar milca')
    # plt.imshow(covar_milca)

    # from scipy.ndimage import gaussian_filter
    # data_90, data_150, beam_handler_90, beam_handler_150 = extract_maps.extract_act_maps_single(inputs, inputs['dec'], inputs['ra'], inputs['map_radius'], 30 * cds.arcsec, inputs['deconvolution_map_radius'], inputs['deconvolve_cmb_lmax'], verbose=True, even_maps=True)
    # plt.figure('data 90')
    # plt.imshow(gaussian_filter(data_90, 2))
    # plt.figure('data 150')
    # plt.imshow(gaussian_filter(data_150, 2))
    # plt.show()
    
    data_milca, beam_handler_milca = extract_maps.extract_milca_maps_single(inputs, inputs['dec'], inputs['ra'], inputs['map_radius'], 10/3 * cds.arcmin, verbose=True, even_maps=True)
    plt.imshow(data_milca)
    plt.show()


if __name__ == "__main__":
    
    freq = 143e9
    T_cmb = 2.726
    x = scipy.constants.h * freq / (scipy.constants.k * T_cmb)
    f_x = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4
    print(f_x)

    milca_y_fpath = '/home/harry/clustergnfwfit_package/data/COM_CompMap_YSZ_R2.02/nilc_ymaps.fits'
    hp_milca, header = hp.fitsfunc.read_map(milca_y_fpath, hdu=1, field=0, memmap=True, h=True)
    # get beam, 10/3 arcmin
    beam_handler_milca = beam_utils.BeamHandlerGaussian(10/60, (10/3) * 60, 9)

    plt.figure('milca beam')
    plt.imshow(beam_handler_milca.beam_map)
    plt.show()

    r_x = 330
    r_y = 160
    r_z = np.sqrt(r_x * r_y)
    offset_x, offset_y = 0, 0
    map_size = 6

    timer = time.time()
    gnfw_s_xy_sqr = ellipsoid_model.interp_gnfw_s_xy_sqr(1, r_x, r_y, r_z, 260)
    print('interp eval', time.time() - timer)

    beam_handler_milca = beam_utils.BeamHandlerGaussian(10/60, 10, 151)
    plt.figure('beam handler 10 arcsec res')
    plt.imshow(beam_handler_milca.beam_map)
    plt.show()
    
    timer = time.time()

    model_no_c_10_correct = ellipsoid_model.eval_pixel_centers_use_interp(gnfw_s_xy_sqr, 0, r_x, r_y, 10, offset_x, offset_y,
                                                                        (map_size * 20) + beam_handler_milca.get_pad_pixels(), (map_size * 20)  + beam_handler_milca.get_pad_pixels())
    # print(time.time() - timer)
    # plt.figure('pre convolve')
    # plt.imshow(model_no_c_10_correct)
    model_no_c_10_correct = beam_handler_milca.convolve2d(model_no_c_10_correct, cut_padding=True)
    print(time.time() - timer)

    plt.figure('after convolve')
    plt.imshow(model_no_c_10_correct)
    model_no_c_10_correct = ellipsoid_model.rebin_2d(model_no_c_10_correct, (20, 20))

    plt.figure('after rebin')
    plt.imshow(model_no_c_10_correct)
    plt.show()

    deg_dec = -11.7525000000
    deg_ra = 206.878333333
    coords = np.deg2rad([deg_dec, deg_ra])

    map_radius = 10 * cds.arcmin
    res = 10/3 * cds.arcmin
    box_radius = 1.5 * map_radius

    ndmap_milca = extract_maps.enmap_from_healpix_in_radius(hp_milca, (deg_dec, deg_ra), box_radius, res, 'car') * f_x * T_cmb
    plt.figure('milca ndmap')
    plt.imshow(ndmap_milca)
    plt.show()

    # oshape = (41, 41)
    # oshape, owcs = enmap.thumbnail_geometry(shape=oshape, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl')
    sfl_milca = reproject.thumbnails(ndmap_milca, coords, r=map_radius.to(cds.deg).value*utils.degree, res=res.to(cds.arcmin).value*utils.arcmin, proj='sfl', verbose=True)
    plt.figure('sfl milca')
    plt.imshow(sfl_milca)
    plt.show()
    
    # Create the box and use it to select a ndmap
    box = np.deg2rad([[deg_dec - box_radius.to(cds.deg).value, deg_ra - box_radius.to(cds.deg).value], [deg_dec + box_radius.to(cds.deg).value, deg_ra + box_radius.to(cds.deg).value]])
    
    # these are in CAR projection and may not be centered on dec, ra
    ndmap_milca = enmap.read_fits(milca_y_fpath, box=box)[0]

    plt.figure('beam')
    plt.imshow(beam_handler_milca.beam_map)

    plt.figure('milca y')
    plt.imshow(ndmap_milca)    
    plt.show()


    micro_k_map = milca_y * f_x * T_cmb * 1e6