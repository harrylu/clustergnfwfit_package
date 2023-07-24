import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy import units as u
from astropy.units import cds
import emcee

# astropy.cosmology comes before cds.enable()
from astropy.cosmology import FlatLambdaCDM
import mcmc_post
import extract_maps
import covariance_matrix
import mcmc
import mcmc_refit

outputs_folder_path = os.path.join(os.getcwd(), "run_outputs")

def parse_input(fpath):
    # lines are:

    # output folder name
    # object name (for fits file)
    # ra (hours space minutes space seconds)
    # dec (degrees space minutes space seconds)

    # path to redshift (Bolocam X_ray_parameters)
    # path to R2500 (Szemasses_SP_2500)
    # cluster name to index redshift, R2500

    # below: mantz paths
    # Path to directory containing Mantz_Xray_masses, Mantz_Xray_pressures, Mantz_Xray_temperatures (make sure ends with /)
    # mantz_cosmology (leave as standard_cosmology)
    # cluster name to index mantz

    # path to SZpack .so file

    # below: paths to fits files containing data
    # path to ACT 90 brightness
    # path to ACT 90 noise
    # path to ACT 90 beam
    # path to ACT 150 brightness
    # path to ACT 150 noise
    # path to ACT 150 beam
    # path to PLANCK CMB
    # path to Bolocam filtered
    # path to Bolocam noise
    # path to Bolocam transfer function

    # map radius (in arcmin); will fit this map 

    # batch size to use when calculating covariance matrix (int, higher is recommended (1000+))
    # radius in which to pick realizations for covar matrix (float, in degrees, 10 recommended)

    # deconvolution_map_radius (float, in degrees, 0.5 recommended)
    # deconvolve_cmb_lmax (int) (2000 recommended)





    with open(fpath, 'rt', encoding='utf-8') as f:
        try: 
            lines = map(lambda s: s.strip(), f.readlines())
            lines = filter(lambda s: s != '' and (len(s) >= 2 and s[:2] != '//'), lines)
            folder_name = next(lines)
            folder_path = os.path.join(outputs_folder_path, folder_name)

            obj_name = next(lines)

            ra = tuple(map(float, next(lines).split(' ')))
            dec = tuple(map(float, next(lines).split(' ')))

            bolocam_xray_parameters_fpath = next(lines)
            r2500_fpath = next(lines)
            redshift_r2500_cluster_name = next(lines)

            mantz_data_dir = next(lines)
            mantz_cosmology = next(lines)
            mantz_cluster_name = next(lines)

            szpack_fpath = next(lines)

            trans_1_dot_5mm = next(lines)
            spectra_2mm = next(lines)
            skysub_250mHz = next(lines)

            brightness_90 = next(lines)
            full_ivar_90 = next(lines)
            beam_90 = next(lines)
            brightness_150 = next(lines)
            full_ivar_150 = next(lines)
            beam_150 = next(lines)
            bandpasses_act = next(lines)
            cmb = next(lines)
            bolocam_filtered = next(lines)
            bolocam_noise_realizations = next(lines)
            bolocam_transfer = next(lines)

            map_radius = float(next(lines)) * cds.arcmin

            covar_num_samples = int(next(lines))
            covar_batch_size = int(next(lines))
            covar_pick_sample_radius = float(next(lines)) * cds.deg

            deconvolution_map_radius = float(next(lines)) * cds.deg
            deconvolve_cmb_lmax = int(next(lines))
        except StopIteration as e:
            raise RuntimeError("Not enough lines in input file. Are you forgetting a parameter?") from e


    return {'folder_path': folder_path,
            'obj_name': obj_name,

            'dec': dec,
            'ra': ra, 
            
            'bolocam_xray_parameters_fpath': bolocam_xray_parameters_fpath,
            'r2500_fpath': r2500_fpath,
            'redshift_r2500_cluster_name': redshift_r2500_cluster_name,

            'mantz_data_dir': mantz_data_dir,
            'mantz_cosmology': mantz_cosmology,
            'mantz_cluster_name': mantz_cluster_name,

            'szpack_fpath': szpack_fpath,

            'trans_1.5mm': trans_1_dot_5mm,
            '2mm_spectra': spectra_2mm,
            'coadd_clean_lissajous_skysub_250mHz_psdfit_nosig': skysub_250mHz,


            'brightness_90': brightness_90,
            'full_ivar_90': full_ivar_90,
            'beam_90': beam_90,
            'brightness_150': brightness_150,
            'full_ivar_150': full_ivar_150,
            'beam_150': beam_150,
            'bandpasses_act': bandpasses_act,
            'cmb': cmb,
            'bolocam_filtered': bolocam_filtered,
            'bolocam_noise_realizations': bolocam_noise_realizations,
            'bolocam_transfer': bolocam_transfer,

            'map_radius': map_radius,

            'covar_num_samples': covar_num_samples,
            'covar_batch_size': covar_batch_size,
            'covar_pick_sample_radius': covar_pick_sample_radius,

            'deconvolution_map_radius': deconvolution_map_radius,
            'deconvolve_cmb_lmax': deconvolve_cmb_lmax,

            }

# params_dict must have keys 'bolocam_xray_parameters_fpath', 'r2500_fpath', 'redshift_r2500_cluster_name', 'mantz_data_dir', 'cosmology', 'mantz_cluster_name',
# returns redshift, r2500, r500 in arcseconds
def get_redshift_r2500_r500(params_dict):
    cosmo = FlatLambdaCDM(70, .3)
    xray_params = params_dict['bolocam_xray_parameters_fpath']
    r2500_path = params_dict['r2500_fpath']
    redshift_r2500_cluster_name = params_dict['redshift_r2500_cluster_name']

    # get Redshift
    xray_name = np.loadtxt(xray_params, usecols=(0), comments=';', dtype=str).T
    z = np.loadtxt(xray_params, usecols=(1), comments=';', dtype=float).T
    redshift = {k: v for k, v in zip(xray_name, z)}

    # get R2500
    xray_name = np.loadtxt(r2500_path, skiprows=1, usecols=(0), dtype=str).T
    r2500 = np.loadtxt(r2500_path, skiprows=1, usecols=(1), dtype=float).T

    r2500 = {k: v * u.Mpc for k, v in zip(xray_name, r2500)}
    # convert r2500 from Mpc to arcseconds
    for name, r2500_mpc in r2500.items():
        # Mpc / rad
        angdist = cosmo.angular_diameter_distance(redshift[name])
        r2500[name] = (r2500_mpc / angdist).value * (3600 * 180)/np.pi

    # get R500
    cluster = params_dict['mantz_cluster_name']
    cosmology = params_dict['mantz_cosmology']
    data_dir = params_dict['mantz_data_dir']
    r500_m500_path = os.path.join(data_dir, 'Mantz_Xray_masses',
                                cosmology, cluster + '_r500_M500.txt')
    this_xray_r500, this_xray_m500 = np.split(np.loadtxt(r500_m500_path).T, 2, axis=0)

    r500_mpc = np.median(this_xray_r500)

    # convert r500 to arcseconds
    angdist = cosmo.angular_diameter_distance(redshift[redshift_r2500_cluster_name])
    r500_arcsec = (r500_mpc / angdist).value * (3600 * 180)/np.pi

    return redshift[redshift_r2500_cluster_name], r2500[redshift_r2500_cluster_name], r500_arcsec

def show_plots(fpath, show_maps=False, show_blurred_maps=False, show_covariance=False, show_first_fit=False, show_refit=False, show_trace=False, show_model=False):
    inputs = parse_input(fpath)

    data_maps_dir = os.path.join(inputs['folder_path'], 'data_maps')
    data_covar_dir = os.path.join(inputs['folder_path'], 'data_covariance')
    if show_maps:
        mcmc_post.show_npy_plots(data_maps_dir)
        if show_blurred_maps:
            mcmc_post.show_npy_plots(data_maps_dir, figure_name_hook=lambda fname: fname + "_blurred", pre_imshow_hook=lambda arr: gaussian_filter(arr, 2))

    if show_covariance:
        mcmc_post.show_npy_plots(data_covar_dir)

    mcmc_out_dir_path = os.path.join(inputs['folder_path'], 'mcmc_first_fit')
    if show_first_fit:
        mcmc_post.show_pickled_plots(os.path.join(mcmc_out_dir_path, 'corner_plots'))
        if show_trace:
            mcmc_post.show_pickled_plots(os.path.join(mcmc_out_dir_path, 'mcmc_trace'))
    
    mcmc_refit_out_dir_path = os.path.join(inputs['folder_path'], 'mcmc_refit')
    if show_refit:
        mcmc_post.show_pickled_plots(os.path.join(mcmc_refit_out_dir_path, 'corner_plots'))
        if show_trace:
            mcmc_post.show_pickled_plots(os.path.join(mcmc_refit_out_dir_path, 'mcmc_trace'))

    if show_model:
        for file in os.listdir(inputs['folder_path']):
            if file.endswith(".fits"):
                hdul = fits.open(os.path.join(inputs['folder_path'], file))
                plt.figure('model')
                plt.imshow(hdul[0].data)
                break
    plt.show()

def run(fpath, run_first_fit=False, run_refit=False, make_corner=False, make_trace=False, make_fits=False):
    inputs = parse_input(fpath)

    print(f"Will output to directory at {inputs['folder_path']}")
    if not os.path.exists(inputs['folder_path']):
        os.makedirs(inputs['folder_path'])
    # if os.path.exists(inputs['folder_path']):
    #     with os.scandir(inputs['folder_path']) as it:
    #         if any(it):
    #             raise Exception(f"Directory at {inputs['folder_path']} is not empty. Please choose a different directory.")
    # else:
    #     os.makedirs(inputs['folder_path'])


    # get redshift, r2500, r500
    redshift, r2500, r500 = get_redshift_r2500_r500(inputs)
    print(f"Redshift: {redshift}, R2500: {r2500}, R500: {r500}")
    inputs['redshift'] = redshift
    inputs['r500'] = r500
    inputs['r2500'] = r2500

    if run_first_fit or run_refit:
        # get or make maps, covar matrices
        # if os.path.exists(os.path.join(maps_covar_dir, 'act_90_data')):
        #     data_90 = 


        # covar_90, covar_150 = covariance_matrix.get_covar_act(inputs['covar_num_samples'], inputs['covar_batch_size'], inputs,
        #                                                       inputs['dec'], inputs['ra'], inputs['covar_pick_sample_radius'], inputs['map_radius'],
        #                                                       deconvolve_cmb_lmax=inputs['deconvolve_cmb_lmax'], verbose=True, even_maps=True)
        
        
        # covar_90, covar_150 = covariance_matrix.get_covar_act(inputs['covar_num_samples'], inputs['covar_batch_size'], inputs,
        #                                                       inputs['dec'], inputs['ra'], inputs['covar_pick_sample_radius'], inputs['map_radius'],
        #                                                       deconvolve_cmb_lmax=inputs['deconvolve_cmb_lmax'], verbose=True, even_maps=True)
        save_path = 'data/covar_np_saves'

        # np.save(os.path.join(save_path, 'act_90_covar_2000_median'), covar_90)
        # np.save(os.path.join(save_path, 'act_150_covar_2000_median'), covar_150)

        # using median increase chi sq by a ton
        covar_90 = np.load(os.path.join(save_path, 'act_90_covar_30000_reject_3_iqr.npy'))
        covar_150 = np.load(os.path.join(save_path, 'act_90_covar_30000_reject_3_iqr.npy'))

        # covar_bolocam = covariance_matrix.get_covar_bolocam(inputs['bolocam_noise_realizations'], 1000)
        
        covar_bolocam = np.load(os.path.join(save_path, 'bolo_covar_1000.npy'))
        covar_bolocam = np.diag(np.diag(covar_bolocam))
        

        
        data_90, data_150, beam_handler_90, beam_handler_150 = extract_maps.extract_act_maps_single(inputs, inputs['dec'], inputs['ra'], inputs['map_radius'], 30 * cds.arcsec, inputs['deconvolution_map_radius'], inputs['deconvolve_cmb_lmax'], verbose=True, even_maps=True)
        data_bolocam, beam_handler_bolocam = extract_maps.extract_bolocam_map(inputs, inputs['dec'], inputs['ra'], inputs['deconvolution_map_radius'], inputs['deconvolve_cmb_lmax'])
        
        print(f'90, 150 shapes: {data_90.shape}, {data_150.shape}')
        print(f'bolocam shapes: {data_bolocam.shape}')
        
        # save data maps, covar
        data_maps_dir = os.path.join(inputs['folder_path'], 'data_maps')
        data_covar_dir = os.path.join(inputs['folder_path'], 'data_covariance')
        if not os.path.exists(data_maps_dir):
            os.mkdir(data_maps_dir)
        if not os.path.exists(data_covar_dir):
            os.mkdir(data_covar_dir)
        np.save(os.path.join(data_maps_dir, 'act_90_data'), data_90)
        np.save(os.path.join(data_maps_dir, 'act_150_data'), data_150)
        np.save(os.path.join(data_maps_dir, 'bolocam_data'), data_bolocam)
        np.save(os.path.join(data_covar_dir, 'act_90_covariance'), covar_90)
        np.save(os.path.join(data_covar_dir, 'act_150_covariance'), covar_150)
        np.save(os.path.join(data_covar_dir, 'bolocam_covariance'), covar_bolocam)

    mcmc_out_dir_path = os.path.join(inputs['folder_path'], 'mcmc_first_fit')
    if run_first_fit:
        print("Running first fit...")
        if os.path.exists(mcmc_out_dir_path):
            raise FileExistsError(f"Directory at {mcmc_out_dir_path} already exists.\nEither set run_first_fit to False to skip or delete the directory to rerun.")
        mcmc.run_mcmc(*map(np.array, (data_90, data_150, data_bolocam)), beam_handler_90, beam_handler_150, beam_handler_bolocam, covar_90, covar_150, covar_bolocam, inputs['r500'], mcmc_out_dir_path)
    if make_corner:
        print("Making corner plot for first fit...")
        mcmc_post.make_corner(mcmc_out_dir_path, 'backend.h5')
    if make_trace:
        print("Making trace plots for first fit...")
        mcmc_post.make_trace(mcmc_out_dir_path, 'backend.h5', 'lower_bounds.txt', 'upper_bounds.txt')

    mcmc_refit_out_dir_path = os.path.join(inputs['folder_path'], 'mcmc_refit')
    if run_refit:
        print("Running refit...")
        if os.path.exists(mcmc_refit_out_dir_path):
            raise FileExistsError(f"Directory at {mcmc_refit_out_dir_path} already exists.\nEither set run_refit to False to skip or delete the directory to rerun.")
        mcmc.run_mcmc(*map(np.array, (data_90, data_150, data_bolocam)), beam_handler_90, beam_handler_150, beam_handler_bolocam, covar_90, covar_150, covar_bolocam, inputs['r500'], mcmc_refit_out_dir_path, mcmc_out_dir_path)
    if make_corner:
        print("Making corner plot for refit...")
        mcmc_post.make_corner(mcmc_refit_out_dir_path, 'backend.h5')
    if make_trace:
        print("Making trace plots for refit...")
        mcmc_post.make_trace(mcmc_refit_out_dir_path, 'backend.h5', 'lower_bounds.txt', 'upper_bounds.txt')

    print("Making fits file...")
    fits_out_path = os.path.join(inputs['folder_path'], f"{inputs['obj_name']}_gNFW_fit.fits")
    if make_fits:
        mcmc_post.make_fits(fits_out_path, mcmc_out_dir_path, mcmc_refit_out_dir_path, inputs)


if __name__ == "__main__":
    fpath = os.path.join(os.getcwd(), "run_inputs", "MACSJ0025.4.txt")
    run(fpath, make_fits=True)
    print(fits.open('/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4_test_full/MACSJ0025.4_gNFW_fit.fits')[0].header)
    exit()
    
    show_plots(fpath, show_maps=True, show_blurred_maps=True, show_model=True)
    # show_plots(fpath, show_first_fit=True, show_refit=True)
    # run(fpath, run_first_fit=False, run_refit=True, make_corner=True, make_trace=True, make_fits=True)