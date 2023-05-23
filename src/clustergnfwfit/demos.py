
import os.path
import numpy as np
import gnfw_fit_map

def demo_fit():
    """Demonstrates gnfw_fit_map.gnfw_fit_map
    """

    MAP_FITS_DIR = "/home/harry/ClusterGnfwFit/map_fits_files"
    FNAME_BRIGHTNESS_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_map_srcfree.fits'
    FNAME_NOISE_150 = 'act_planck_dr5.01_s08s18_AA_f150_night_ivar.fits'
    FNAME_BRIGHTNESS_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits'
    FNAME_NOISE_90 = 'act_planck_dr5.01_s08s18_AA_f090_night_ivar.fits'
    FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb
    
    # beam of width 17 pixels has smallest values which are within 1% of largest
    BEAM_MAP_WIDTH = 17
    FPATH_BEAM_150 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f150_night_beam.txt"
    FPATH_BEAM_90 = r"/home/harry/ClusterGnfwFit/act_dr5.01_auxilliary/beams/act_planck_dr5.01_s08s18_f090_night_beam.txt"

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
    }
    

    # these fields will vary depending on the cluster
    dec = [-12, -22, -45]  # in degrees, minutes, seconds
    ra = [0, 25, 29.9]     # in hours, minutes, seconds
    #dec = [0, 0, 0]  # in degrees, minutes, seconds
    #ra = [0, 0, 0]     # in hours, minutes, seconds
    # ra = [0, 25, 29.9]
    map_radius = 5  # arcminutes
    R500 = 200  # arcseconds
    init_params = (-22,      -45,     170,    0,  0,      0,      0)
    fixed_params = (False, False, False, False, False, False, False)
    num_processes = 7   # we will use 7 cores
    params, perror = gnfw_fit_map.fit_map(fpath_dict, BEAM_MAP_WIDTH,
                                            dec, ra, map_radius, R500,
                                            init_params, fixed_params,
                                            True, True, num_processes)
    print(params)


from matplotlib import pyplot as plt
import eval_gnfw
import di_utils
from conversions import convert_microkelvin_to_mjysr
from fits_creator import make_fits
def demo_fits_maps_and_di():
    """Demonstrates generating the maps to be put into the fits file as well
    as getting the di value
    """

    # params, perror = gnfw_fit_map,gnfw_fit_map(...)

    # I will set the params manually for this demo
    # based on previous fit done on MACSJ0025.4

    # This part demonstrates making the map that will go into the fits file.
    # The map is centered and 470*470 pixels with 4 arcsecond pixels

    dec = [-12, -22, -45]  # in degrees, minutes, seconds
    ra = [0, 25, 29.9]     # in hours, minutes, seconds

    def hms_to_deg(hours, minutes, seconds):
        return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
    def dms_to_deg(degrees, minutes, seconds):
        return degrees + minutes / 60 + seconds / (60 ** 2)

    decimal_dec = dms_to_deg(*dec)
    decimal_ra = hms_to_deg(*ra)

    R500 = 200
    R2500 = 66.7104
    params = [-2.80218707e+00, -4.67692923e+00,  4.04650453e+02, -2.67669039e-02,
 -3.85902294e-03,  3.94686241e+01,  2.64345529e+01]
    errors = [  1.74265301,   3.04220972, 106.55782932,   0.31058793,   0.31098361,
   4.10405139,   5.36751085]
    P0_150, P0_90, RS, _, _, c_150, c_90 = params
    err_150, err_90, _, _, _, _, _ = errors
    # we can avoid having to call make_fits_grid twice by calling it once
    # and then multiplying it by the ratio to get the other before adding the constants
    gnfw_fits_150 = eval_gnfw.make_fits_grid(P0_150, RS, R500, 4, 1e-2, num_processes=4)
    gnfw_fits_90 = gnfw_fits_150 * (P0_90/P0_150)

    # now, we can add the additive constants
    # nevermind, offsets are just artifacts, don't add
    # gnfw_fits_150 += c_150
    # gnfw_fits_90 += c_90

    # map is evaluated in microKelvin so convert to MJY*SR
    gnfw_fits_150 = convert_microkelvin_to_mjysr(gnfw_fits_150, 150)
    gnfw_fits_90 = convert_microkelvin_to_mjysr(gnfw_fits_90, 90)
    # temporary, for testing purposes, so we don't have to recompute every time
    np.save('normalized_gnfw_model.npy', gnfw_fits_150)

    # now, get di values
    di_150 = di_utils.get_R2500_avg(gnfw_fits_150, 4, R2500)
    di_90 = di_utils.get_R2500_avg(gnfw_fits_90, 4, R2500)
    # normalized model
    normalized = gnfw_fits_150 / di_150
    # normalized = gnfw_fits_90 /= di_90

    # now, we have maps of the fits where the average value within R2500 is 1
    # we also have the di value

    # double check
    print('di 150', di_150)
    print('di 90', di_90)
    print('avg within R2500 for normalized', di_utils.get_R2500_avg(normalized, 4, R2500))

    # and, we can get an approximation of the sigma di
    sigma_150 = di_utils.calc_sigma_di(150, err_150, P0_150, di_150)
    sigma_90 = di_utils.calc_sigma_di(90, err_90, P0_90, di_90)
    print('sigma di 150', sigma_150)
    print('sigma di 90', sigma_90)

    # write to fits file
    fpath = 'new1.fits'
    make_fits(fpath, decimal_ra, decimal_dec, normalized, di_90, sigma_90, di_150, sigma_150)

    # show normalized map
    print(np.sum(np.abs(gnfw_fits_90/di_90 - gnfw_fits_150/di_150)))
    plt.figure(0)
    plt.title('fits data')
    plt.imshow(normalized)
    

if __name__ == "__main__":
   # demo_fit()
   demo_fits_maps_and_di()

