import numpy as np
from pixell import enmap, reproject, utils
import healpy as hp
from astropy.io import fits
from scipy import fft
from astropy.modeling.functional_models import Gaussian2D
from astropy.units import cds
cds.enable()

import beam_utils
import deconvolution

def enmap_from_healpix_in_radius(hp_map, coords, radius, res, proj):
    # coords is tuple (dec, ra) in degrees
    deg_dec, deg_ra = coords
    box_radius_deg = radius.to(cds.deg).value
    box = np.deg2rad([[deg_dec - box_radius_deg, deg_ra - box_radius_deg], [deg_dec + box_radius_deg, deg_ra + box_radius_deg]])
    # enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape reads +x, +y
    cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=res.to(cds.arcmin).value * utils.arcmin, proj=proj, force=True)
    res = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
                                            ncomp=1, unit=1e-6, rot='gal,equ')[0]
    return res
    
# enmaps from healpix tile by tile to get around memory limits
def enmap_from_healpix_tiles(hp_map, ll_pos, tile_shape, ntile, res, proj='car'):
    # ll_pos is tuple (dec, ra) of lower left corner (0, 0) in degrees (will be wcs of resulting enmap)
    # res in astropy units
    # tile_shape is tuple (height, width)
    # ntile is tuple (# tiles vertical, # tiles horizontal)
    # reads in starting from ll_pos, then ntiles upwards / rightwards
    dec, ra = ll_pos
    ll_pos_rads = np.deg2rad(ll_pos)
    box = np.deg2rad([[dec, ra], [dec + res.to(cds.deg).value * ntile[0], ra + res.to(cds.deg).value * ntile[1]]])
    shape, wcs = enmap.geometry(pos=box, res=res.to(cds.arcmin).value * utils.arcmin, proj=proj, force=True)
    ll_pos_pixels = np.array(enmap.sky2pix(shape, wcs, ll_pos_rads))
    pos_list = []

    for y in range(ntile[0]):
        for x in range(ntile[1]):
            ll_offset = np.array([y * tile_shape[0], x * tile_shape[1]])
            pix = ll_pos_pixels + ll_offset
            # pos_list.append(enmap.pix2sky(shape, wcs, pix))
            # stamps actually takes pos in pixel coords
            pos_list.append(pix)

    stamps = enmap.stamps(enmap.ndmap(np.empty((1, 1)), wcs), pos_list, tile_shape, aslist=True)
    tiles = []
    tile_row = []
    for stamp in stamps:
        stamp = reproject.enmap_from_healpix(hp_map, stamp.shape, stamp.wcs, ncomp=1, unit=1e-6, rot='gal,equ')[0]
        tile_row.append(stamp)
        if len(tile_row) == ntile[1]:
            tiles.append(tile_row)
            tile_row = []
    return enmap.tile_maps(tiles)

def extract_act_maps_for_covar(num_maps, fpath_dict, dec, ra, pick_sample_radius, map_radius, deconvolve_cmb_lmax, verbose, even_maps):
    # radiuses must be in astropy units
    
    def hms_to_deg(hours, minutes, seconds):
        return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
    def dms_to_deg(degrees, minutes, seconds):
        return degrees + minutes / 60 + seconds / (60 ** 2)

    decimal_dec = dms_to_deg(*dec)
    decimal_ra = hms_to_deg(*ra)

    # I_STOKES_INP is column (field) 5
    hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
    # get CMB beam
    hdul = fits.open(fpath_dict['cmb'])
    beam_hdu = hdul[2]
    Bl = list(beam_hdu.columns['INT_BEAM'].array)

    # diameter of 17 pixels has pixels at < 1% of highest
    beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], 17)
    beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], 17)    

    # x arcmins = x/60 deg
    # add some positive 2 arcmin to map_radius so we dont lose data when we reproject
    box_deg_r = (pick_sample_radius.to(cds.arcmin) + map_radius.to(cds.arcmin) + 2 * cds.arcmin).to(cds.deg).value

    # Create the box and use it to select a submap enmap
    box = np.deg2rad([[decimal_dec - box_deg_r, decimal_ra - box_deg_r], [decimal_dec + box_deg_r, decimal_ra + box_deg_r]])

    # resolution is 30 arcseconds, will use later
    res = 30 * cds.arcsec

    # these are in CAR projection (may not be centered on dec, ra)
    enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
    enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
    
    # get sample offsets in decimal degrees
    sample_coords = (np.random.random((num_maps, 2)) * (2 * pick_sample_radius) - pick_sample_radius).to(cds.deg).value
    # add dec, ra
    sample_coords[:, 0] += decimal_dec
    sample_coords[:, 1] += decimal_ra
    
    
    even_maps = True
    # Need to do some stuff if we want even maps
    if even_maps:
        half_pixel_deg = (res / 2).to(cds.deg).value
        sfl_90 = reproject.thumbnails(enmap_90, np.deg2rad(sample_coords + half_pixel_deg), r=map_radius.to(cds.arcmin).value * utils.arcmin, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)
        sfl_150 = reproject.thumbnails(enmap_150, np.deg2rad(sample_coords + half_pixel_deg), r=map_radius.to(cds.arcmin).value * utils.arcmin, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)
        sfl_90 = sfl_90[:, 1:, 1:]
        sfl_150 = sfl_150[:, 1:, 1:]
    else:
        sfl_90 = reproject.thumbnails(enmap_90, np.deg2rad(sample_coords), r=map_radius.to(cds.arcmin).value * utils.arcmin, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)
        sfl_150 = reproject.thumbnails(enmap_150, np.deg2rad(sample_coords), r=map_radius.to(cds.arcmin).value * utils.arcmin, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)

    # ntiles = 1
    # res = 30 * cds.arcsec
    # tile_shape = np.ceil(2 * box_deg_r / res.to(cds.deg).value / ntiles)
    # tile_shape = (tile_shape, tile_shape)
    # ndmap_cmb = enmap_from_healpix_tiles(hp_map, (decimal_dec - box_deg_r, decimal_ra - box_deg_r), tile_shape, (ntiles, ntiles), res, proj='car')
    ndmap_cmb = enmap_from_healpix_in_radius(hp_map, (decimal_dec, decimal_ra), box_deg_r * cds.deg, res, 'car')
    print(f"Enmapped from healpix with shape of {ndmap_cmb.shape}")
    ndmaps_deconvolved_cmb = deconvolution.get_deconvolved_map(np.array(sfl_90.shape[1:]) + beam_handler_90.get_pad_pixels(), ndmap_cmb, Bl, sample_coords, 0.5 * cds.deg, res=res, lmax=deconvolve_cmb_lmax, proj='sfl')

    # convolve deconvolved cmb with 90 GHz, 150 GHz beams
    deconvolved_cmb_90 = [beam_handler_90.convolve2d(ndmap) for ndmap in ndmaps_deconvolved_cmb]
    deconvolved_cmb_150 = [beam_handler_150.convolve2d(ndmap) for ndmap in ndmaps_deconvolved_cmb]
    

    # subtract cmb
    sfl_90_cmb_subtracted = sfl_90 - np.array(deconvolved_cmb_90)
    sfl_150_cmb_subtracted = sfl_150 - np.array(deconvolved_cmb_150)

    # for i, (sfl_90_cmb_subtracted, sfl_90, deconvolved_cmb_90, sfl_150_cmb_subtracted, sfl_150, deconvolved_cmb_150) in enumerate(zip(sfl_90_cmb_subtracted, sfl_90, deconvolved_cmb_90, sfl_150_cmb_subtracted, sfl_150, deconvolved_cmb_150)):
    #     plt.figure(i)
    #     f, axs = plt.subplots(2, 3)
    #     import scipy.ndimage
    #     from matplotlib import cm
    #     axs[0][0].imshow(scipy.ndimage.gaussian_filter(sfl_90_cmb_subtracted, (1.5, 1.5)), cmap=cm.coolwarm, vmin=-100, vmax=100)
    #     axs[0][1].imshow(scipy.ndimage.gaussian_filter(sfl_90, (1.5, 1.5)), cmap=cm.coolwarm, vmin=-100, vmax=100)
    #     axs[0][2].imshow(scipy.ndimage.gaussian_filter(deconvolved_cmb_90, (1.5, 1.5)), cmap=cm.coolwarm, vmin=-100, vmax=100)
    #     axs[1][0].imshow(scipy.ndimage.gaussian_filter(sfl_150_cmb_subtracted, (1.5, 1.5)), cmap=cm.coolwarm, vmin=-100, vmax=100)
    #     axs[1][1].imshow(scipy.ndimage.gaussian_filter(sfl_150, (1.5, 1.5)), cmap=cm.coolwarm, vmin=-100, vmax=100)
    #     axs[1][2].imshow(scipy.ndimage.gaussian_filter(deconvolved_cmb_150, (1.5, 1.5)), cmap=cm.coolwarm, vmin=-100, vmax=100)
    # plt.show()

    return sfl_90_cmb_subtracted, sfl_150_cmb_subtracted


'''if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    MAP_FITS_DIR = "/home/harry/clustergnfwfit_package/data/map_fits_files"
    FNAME_CMB = 'COM_CMB_IQU-commander_2048_R3.00_full.fits'   # the healpix cmb

    hp_path = os.path.join(MAP_FITS_DIR, FNAME_CMB)
    hp_map, header = hp.fitsfunc.read_map(hp_path, field=5, hdu=1, memmap=True, h=True)

    dec = [-12, -22, -45]  # in degrees, minutes, seconds
    ra = [0, 25, 29.9]     # in hours, minutes, seconds
    def hms_to_deg(hours, minutes, seconds):
        return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
    def dms_to_deg(degrees, minutes, seconds):
        return degrees + minutes / 60 + seconds / (60 ** 2)

    decimal_dec = dms_to_deg(*dec)
    decimal_ra = hms_to_deg(*ra)
    map_radius = 30 * cds.arcmin    # 30 arcmins -> degs
    
    box_radius_deg = map_radius.to(cds.deg).value    # read in a little extra so reprojection doesnt lose data
    box = np.deg2rad([[decimal_dec - box_radius_deg, decimal_ra - box_radius_deg], [decimal_dec + box_radius_deg, decimal_ra + box_radius_deg]])
    # enmap_90's wcs stores the (0, 0) of the ndmap and reproject.enmap_from_healpix's shape reads +x, +y
    # so we have to make our own wcs that is 1deg x 1deg and centered at our dec, ra
    cmb_shape, cmb_wcs = enmap.geometry(pos=box, res=1/2 * utils.arcmin, proj='car', force=True)
    enmap_cmb = reproject.enmap_from_healpix(hp_map, cmb_shape, cmb_wcs, 
                                            ncomp=1, unit=1e-6, rot='gal,equ')[0]
    plt.figure('from regular')
    plt.imshow(enmap_cmb)

    ntiles = 3
    res = 30 * cds.arcsec
    shape = np.ceil(2 * map_radius.to(cds.arcmin) / res.to(cds.arcmin) / ntiles)
    shape = (shape, shape)
    print(shape)
    ndmap_car = enmap_from_healpix_tiles(hp_map, (decimal_dec - map_radius.to(cds.deg).value, decimal_ra - map_radius.to(cds.deg).value), shape, (ntiles, ntiles), res)
    plt.figure('from tiles')
    plt.imshow(ndmap_car)

    pad_car = np.zeros_like(enmap_cmb)
    pad_car[:ndmap_car.shape[0], :ndmap_car.shape[1]] = ndmap_car

    plt.figure('difference')
    plt.imshow(np.abs(pad_car - enmap_cmb), vmin=0, vmax=0.1)

    plt.show()'''

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
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
    map_radius = 10 * cds.arcmin # arcminutes
    R500 = 200  # arcseconds

    maps_90, maps_150 = extract_act_maps_for_covar(10, fpath_dict, dec, ra, 20 * cds.deg, map_radius, 2000, True, True)

    for i, (m_90, m_150) in enumerate(zip(maps_90, maps_150)):
        plt.figure(i)
        f, axs = plt.subplots(2, 1)
        axs[0].imshow(m_90, cmap=cm.coolwarm, vmin=-100, vmax=100)
        axs[1].imshow(m_150, cmap=cm.coolwarm, vmin=-100, vmax=100)
    plt.show()


def extract_maps(fpath_dict,
                dec, ra, map_radius,
                deconvolve_cmb_lmax=2000, include_bolocam=True, verbose=False, even_maps=True):
    """Extracts specified region in ACTPlanck data with cmb subtracted.
    ACTPlanck maps are here: https://lambda.gsfc.nasa.gov/product/act/actpol_dr5_coadd_maps_get.html
    Cmb deconvolution is done via division in Fourier space with low pass filtering.
    Cmb maps are here: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/previews/COM_CMB_IQU-commander_2048_R3.00_full/
    Uses I_STOKES_INP ^.
    Bolocam maps are here: https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/bolocam/bolocam.html

    Args:
        fpath_dict (dict): contains file paths; should contain keys:
            'brightness_150': path to brightness 150 fits file,
            'noise_150': 'path to 150 GHz noise fits file,
            'brightness_90': path to 90 GHz brightness fits file,
            'noise_90', 'path to 90 GHz noise fits file,
            'cmb', 'beam_150', 'beam_90': self-explanatory
            Optional:
            'bolocam_filtered': path to filtered bolocam fits file,
            'bolocam_noise': path to bolocam RMS noise fits file,
            'bolocam_transfer': path to bolocam transfer fits file,
        dec (tuple): declination in (degrees, minutes, seconds)
        ra (tuple): right ascension in (hours, minutes, seconds)
        map_radius (float): in arcminutes; radial width of the map that will be extracted
        deconvolve_cmb_lmax (int): lmax to keep in deconvolved cmb 
        verbose (bool, optional): Whether to log to console. Defaults to False.

    Notes:
        The extracted maps will be centered at the (dec, ra) and so will always have to be an odd-numbered shape.
        Will raise error otherwise.

    Returns:
        6 length tuple
        Elements 0, 1: SFL-reprojected maps of specified map_radius 90 GHz, 150 GHz, respectively
        Elements 2, 3: One sigma errors on the maps, 90 GHz, 150 GHz respectively
        Elements 4, 5: BeamHandler instances 90 GHz, 150 GHz, respectively
        Elements 6, 7: bolocam_map, bolocam_beam_handler or None, None if include_bolocam=False


    """
    
    def hms_to_deg(hours, minutes, seconds):
        return (hours + minutes / 60 + seconds / (60 ** 2)) * 15
    def dms_to_deg(degrees, minutes, seconds):
        return degrees + minutes / 60 + seconds / (60 ** 2)

    decimal_dec = dms_to_deg(*dec)
    decimal_ra = hms_to_deg(*ra)
    

    # x arcmins = x/60 deg
    # add some positive 2 arcmin to map_radius so we dont lose data when we reproject
    deg_r = (map_radius + 2) / 60

    # Create the box and use it to select a submap enmap
    box = np.deg2rad([[decimal_dec - deg_r, decimal_ra - deg_r], [decimal_dec + deg_r, decimal_ra + deg_r]])

    # resolution is 30 arcseconds, will use later
    res = 30 * cds.arcsec

    # these are in CAR projection (may not be centered on dec, ra)
    enmap_90 = enmap.read_fits(fpath_dict['brightness_90'], box=box)[0]
    enmap_90_noise = enmap.read_fits(fpath_dict['noise_90'], box=box)[0]
    enmap_150 = enmap.read_fits(fpath_dict['brightness_150'], box=box)[0]
    enmap_150_noise = enmap.read_fits(fpath_dict['noise_150'], box=box)[0]
    
    radius = map_radius*utils.arcmin
    even_maps = True
    # Need to do some stuff if we want even maps
    if even_maps:
        half_pixel_deg = res.to(cds.deg).value
        coords = [np.deg2rad([decimal_dec + half_pixel_deg, decimal_ra + half_pixel_deg])]
        sfl_90 = reproject.thumbnails(enmap_90, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150 = reproject.thumbnails(enmap_150, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_90 = sfl_90[1:, 1:]
        sfl_90_noise = sfl_90_noise[1:, 1:]
        sfl_150 = sfl_150[1:, 1:]
        sfl_150_noise = sfl_150_noise[1:, 1:]
    else:
        # for odd maps
        # reproject to sfl thumbnails (definitely centered on dec, ra)
        coords = [np.deg2rad([decimal_dec, decimal_ra])]
        sfl_90 = reproject.thumbnails(enmap_90, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_90_noise = reproject.thumbnails_ivar(enmap_90_noise, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150 = reproject.thumbnails(enmap_150, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        sfl_150_noise = reproject.thumbnails_ivar(enmap_150_noise, coords, r=radius, res=res.to(cds.arcmin).value * utils.arcmin, proj='sfl', verbose=verbose)[0]
        print(f"ACTPlanck SFL WCS: {sfl_90.wcs}")
    

    assert sfl_90.shape[0] == sfl_90.shape[1], f"Sfl 90 axis length mismatch: {sfl_90.shape}"
    assert sfl_90.shape == sfl_150.shape

    # I_STOKES_INP is column (field) 5
    hp_map, header = hp.fitsfunc.read_map(fpath_dict['cmb'], field=5, hdu=1, memmap=True, h=True)
    # extract 1 degree x 1 degree map (slightly larger so that there is center pixel)
    cmb_radius_deg = 0.503 * cds.deg
    # get CMB beam
    hdul = fits.open(fpath_dict['cmb'])
    beam_hdu = hdul[2]
    Bl = list(beam_hdu.columns['INT_BEAM'].array)

    # diameter of 17 pixels has pixels at < 1% of highest
    beam_handler_150 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_150'], 17)
    beam_handler_90 = beam_utils.BeamHandlerACTPol(fpath_dict['beam_90'], 17)

    ndmap_cmb = enmap_from_healpix_in_radius(hp_map, (decimal_dec, decimal_ra), cmb_radius_deg, 30 * cds.arcsec, 'car')
    enmap_deconvolved_cmb = deconvolution.get_deconvolved_map(np.array(sfl_90.shape) + beam_handler_90.get_pad_pixels(), ndmap_cmb, Bl, [(decimal_dec, decimal_ra)], cmb_radius_deg, res=res, lmax=deconvolve_cmb_lmax, proj='sfl')

    # convolve deconvolved cmb with 90 GHz, 150 GHz beams
    deconvolved_cmb_90 = beam_handler_90.convolve2d(enmap_deconvolved_cmb)
    deconvolved_cmb_150 = beam_handler_150.convolve2d(enmap_deconvolved_cmb)

    # subtract cmb
    sfl_90_cmb_subtracted = sfl_90 - deconvolved_cmb_90
    sfl_150_cmb_subtracted = sfl_150 - deconvolved_cmb_150 

    import matplotlib.pyplot as plt
    from matplotlib import cm
    # plt.figure('deconvolved cmb 90')
    # plt.imshow(deconvolved_cmb_90, cmap=cm.coolwarm, vmin=-100, vmax=100)

    def ivar_to_sigma(x): return np.sqrt(1 / x)
    err_90 = ivar_to_sigma(sfl_90_noise)
    err_150 = ivar_to_sigma(sfl_150_noise)

    # get bolocam maps
    if include_bolocam == True:
        #read FITS header
        header = fits.open(fpath_dict['bolocam_filtered'])[0].header

        # wcs incorrect when read in this way
        enmap_bolocam_filtered = enmap.read_fits(fpath_dict['bolocam_filtered'])
        # fix WCS
        enmap_bolocam_filtered.wcs.wcs.cdelt = [float(header['CD1_1']), float(header['CD2_2'])]
        print(f'Bolocam WCS: {enmap_bolocam_filtered.wcs}')

        # beam is approx gaussian, fwhm in degrees
        bolocam_beam_fwhm = header['BMAJ']
        bolocam_beam_handler = beam_utils.BeamHandlerBolocam(bolocam_beam_fwhm, 11)

        # 20 arcsecond resolution
        enmap_deconvolved_cmb = deconvolution.get_deconvolved_map(np.array(enmap_bolocam_filtered.shape) + bolocam_beam_handler.get_pad_pixels(), ndmap_cmb, Bl, decimal_dec, decimal_ra, cmb_radius_deg, res=20 * cds.arcsec, lmax=deconvolve_cmb_lmax, proj='sfl')
        assert enmap_deconvolved_cmb.shape[0] % 2 == 0
        assert enmap_deconvolved_cmb.shape[0] == enmap_deconvolved_cmb.shape[1]

        # convolve CMB with Bolocam psf
        enmap_deconvolved_cmb = bolocam_beam_handler.convolve2d(enmap_deconvolved_cmb)
        
        # apply hanning
        hanning = np.outer(np.hanning(42), np.hanning(42))
        hanning /= np.mean(hanning)
        enmap_deconvolved_cmb *= hanning

        # filter cmb
        transfer_function_hdul = fits.open(fpath_dict['bolocam_transfer'])
        signal_transfer_function_fft = transfer_function_hdul[0].data + 1j * transfer_function_hdul[1].data
        enmap_deconvolved_cmb = np.real(fft.ifft2(fft.fft2(enmap_deconvolved_cmb) * signal_transfer_function_fft))

        bolocam_map = enmap_bolocam_filtered - enmap_deconvolved_cmb

        bolocam_err = fits.open(fpath_dict['bolocam_noise'])[0].data

        
    else:
        bolocam_map = None
        bolocam_err = None
        bolocam_beam_handler = None

    return sfl_90_cmb_subtracted, sfl_150_cmb_subtracted, err_90, err_150, beam_handler_90, beam_handler_150, bolocam_map, bolocam_err, bolocam_beam_handler
