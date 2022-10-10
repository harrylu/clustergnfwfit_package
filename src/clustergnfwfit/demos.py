
import os.path
from . import gnfw_fit_map

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
    fixed_params = (False, False, True, False, False, False, False)
    num_processes = 7   # we will use 7 cores
    params, perror = gnfw_fit_map.fit_map(fpath_dict, BEAM_MAP_WIDTH,
                                            dec, ra, map_radius, R500,
                                            init_params, fixed_params,
                                            True, True, num_processes)
    print(params)


from matplotlib import pyplot as plt
from .import eval_gnfw
from . import di_utils
from .conversions import convert_microkelvin_to_mjysr
def demo_fits_maps_and_di():
    """Demonstrates generating the maps to be put into the fits file as well
    as getting the di value
    """

    # params, perror = gnfw_fit_map,gnfw_fit_map(...)

    # I will set the params manually for this demo
    # based on previous fit done on MACSJ0025.4

    # This part demonstrates making the map that will go into the fits file.
    # The map is centered and 470*470 pixels with 4 arcsecond pixels

    R500 = 200
    R2500 = 66.7104
    params = [-2.20013204e+01, -4.52118956e+01,  1.70000000e+02, -1.78802476e-01,
 -6.39230798e-02,  3.46171546e+01,  2.11105372e+01]
    errors = [3.55957792, 6.02193125, 0.,         0.25361107, 0.26156104, 3.30316066,
 4.27369931]
    P0_150, P0_90, RS, _, _, c_150, c_90 = params
    err_150, err_90, _, _, _, _, _ = errors
    # we can avoid having to call make_fits_grid twice by calling it once
    # and then multiplying it by the ratio to get the other before adding the constants
    gnfw_fits_150 = eval_gnfw.make_fits_grid(P0_150, RS, R500, 4, 1e-2, num_processes=4)
    gnfw_fits_90 = gnfw_fits_150 * (P0_90/P0_150)

    # now, we can add the additive constants
    gnfw_fits_150 += c_150
    gnfw_fits_90 += c_90

    # map is evaluated in microKelvin so convert to MJY*SR
    gnfw_fits_150 = convert_microkelvin_to_mjysr(gnfw_fits_150, 150)
    gnfw_fits_90 = convert_microkelvin_to_mjysr(gnfw_fits_90, 90)

    # now, get di values
    di_150 = di_utils.get_R2500_avg(gnfw_fits_150, 4, R2500)
    di_90 = di_utils.get_R2500_avg(gnfw_fits_90, 4, R2500)
    gnfw_fits_150 /= di_150
    gnfw_fits_90 /= di_90

    # now, we have maps of the fits where the average value withing R2500 is 1
    # we also have the di value

    print('di 150', di_150)
    print('di 90', di_90)
    print('avg within R2500 for 150', di_utils.get_R2500_avg(gnfw_fits_150, 4, R2500))
    print('avg within R2500 for 90', di_utils.get_R2500_avg(gnfw_fits_90, 4, R2500))

    # and, we can get a Mickey Mouse approximation of the error on di
    print('sigma di 150', di_utils.calc_sigma_di(150, err_150, P0_150, di_150))
    print('sigma di 90', di_utils.calc_sigma_di(90, err_90, P0_90, di_90))

    # show maps
    plt.figure(0)
    plt.title('fits 150')
    plt.imshow(gnfw_fits_150)
    plt.figure(1)
    plt.title('fits 90')
    plt.imshow(gnfw_fits_90)
    plt.show()
