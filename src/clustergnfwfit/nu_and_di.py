import os
import numpy as np
import szpack_bindings
import scipy as sp
import scipy.io
import scipy.interpolate
from pixell import enmap
import os

'''
def rsz_get_mass_weighted_xray_temperature():
    # use the MCMC files produced for the pressure profile project
    redshift = 0.451
    cluster = 'rxj1347.5-1145'
    cosmology = 'standard_cosmology'
    data_dir = 'data/planck/'

    # get the mass and scale radius
    r500_m500_path = os.path.join(data_dir, 'Mantz_Xray_masses',
                                cosmology, cluster + '_r500_M500.txt')
    this_xray_r500, this_xray_m500 = np.split(np.loadtxt(r500_m500_path).T, 2, axis=0)

    r500 = np.median(this_xray_r500)
    r2500 = 0.71
    m500 = np.median(this_xray_m500) * 1.e-14

    # get the pressure profile values
    conversions_path = os.path.join(data_dir, 'Mantz_Xray_pressures',
                                    cosmology, cluster + '_conversions.txt')
    xray_conversions = []
    with open(conversions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' ').split(' ')
            if line[0][0] != '-' and len(line) == 3:
                xray_conversions.append(float(line[1]))
    xray_conversions = np.array(xray_conversions)

    pressure_path = os.path.join(data_dir, 'Mantz_Xray_pressures',
                                 cluster + '_pressure')
    pressure_dat_path = pressure_path + '.dat'
    pressure_float_data = np.loadtxt(pressure_dat_path, skiprows=1, usecols=(0, 4, 5, 6)).T
    xray_radius, xray_pressure, xray_pressure_lo, xray_pressure_hi = np.split(pressure_float_data, 4, axis=0)
    xray_radius = xray_radius[0]    # get 1d np array
    pressure_txt_path = pressure_path + '.txt'
    xray_pressure = np.loadtxt(pressure_txt_path).T
    xray_use = np.loadtxt(pressure_dat_path, skiprows=1, usecols=(3), dtype=str).T
    
    valid_xray = np.nonzero((xray_use == 'TRUE'))[0]
    xray_pressure = xray_pressure[valid_xray]
    # xray_radius = xray_radius[valid_xray] * xray_conversions[1] / r500[0]
    xray_radius = xray_radius[valid_xray] * xray_conversions[1] / r2500

    # get the temperature profile values
    temperature_path = os.path.join(data_dir, 'Mantz_Xray_temperatures',
                                    cosmology, cluster + '_temperature.txt')
    xray_t = np.loadtxt(temperature_path).T
    xray_temperature = xray_t[valid_xray]

    # create a data cube to integrate
    dx = 0.01
    max_x = 1
    max_z = np.max(xray_radius)
    x1d = np.arange(np.round(max_x*2/dx)+1) * dx
    x1d = x1d - np.mean(x1d)
    z1d = np.arange(np.round(max_z*2/dx)+1) * dx
    z1d = z1d - np.mean(z1d)
    nx = np.size(x1d)
    nz = np.size(z1d)
    x3d = np.zeros((nx, nx, nz))
    y3d = np.zeros((nx, nx, nz))
    z3d = np.zeros((nx, nx, nz))
    for iy in range(nx):
        for iz in range(nz):
            x3d[:, iy, iz] = x1d
    for ix in range(nx):
        for iz in range(nz):
            y3d[ix, :, iz] = x1d
    for ix in range(nx):
        for iy in range(nx):
            x3d[ix, iy, :] = z1d
    r3d = np.sqrt(x3d**2 + y3d**2 + z3d**2)
    r2d = np.sqrt(x3d**2 + y3d**2)
    valid = (r2d <= 1)  # not actually translation from IDL, but easier for Python

    un_normalized_density = xray_pressure / xray_temperature
    n_jk = 100
    mw_temperature = np.zeros((n_jk))
    pw_temperature = np.zeros((n_jk))
    for i_jk in range(n_jk):
        temperature_3d = np.interp(r3d, xray_radius, xray_temperature[:, i_jk])
        density_3d = np.interp(r3d, xray_radius, un_normalized_density[:, i_jk])
        pressure_3d = np.interp(r3d, xray_radius, xray_pressure[:, i_jk])

        mw_temperature[i_jk] = np.sum(temperature_3d[valid] * density_3d[valid]) / np.sum(density_3d[valid])
        pw_temperature[i_jk] = np.sum(temperature_3d[valid] * pressure_3d[valid]) / np.sum(pressure_3d[valid])

    # correct for bias of 0.09 +- 0.13 found in Wan et al.
    print(f"T_mw = {np.median(mw_temperature) / 1.09}")
    print(f"sigma_T = {np.sqrt((np.std(mw_temperature) / 1.09)**2 + (np.median(mw_temperature) / 1.09 * 0.13)**2)}")

    # correct for bias of 0.09 +- 0.13 found in Wan et al.
    print(f"T_pw = {np.median(pw_temperature) / 1.09}")
    print(f"sigma_T = {np.sqrt((np.std(pw_temperature) / 1.09)**2 + (np.median(pw_temperature) / 1.09 * 0.13)**2)}")

    # return
    # compute the mass-weighted temperature
    un_normalized_density = xray_pressure / xray_temperature
    n_jk = np.size(xray_pressure[0])
    weight = np.outer(xray_radius**2 * un_normalized_density[:, 0], np.full((n_jk), 1.))
    numerator = np.sum(xray_temperature * weight, axis=0)
    denominator = np.sum(weight, axis=0)
    mass_weighted_temperature = numerator / denominator
    
    # get the statistics
    print(f"T_mw = {np.median(mass_weighted_temperature)}")
    print(f"sigma_T = {np.std(mass_weighted_temperature)}")
    
    # correct for bias of 0.09 +- 0.13 found in Wan et al.
    print(f"T_mw = {np.median(mass_weighted_temperature) / 1.09}")
    print(f"sigma_T = {np.sqrt((np.std(mass_weighted_temperature) / 1.09)**2 + (np.median(mass_weighted_temperature) / 1.09 * 0.13)**2)}")

    print(f"max R/R500 = {np.max(xray_radius)}")
'''
    
# translated from Jack's IDL code with modifications
# params_dict should have keys 'mantz_data_dir', 'mantz_cluster_name', 'mantz_cosmology', 'redshift'
def get_weighted_xray_temperature(params_dict):    # redshift, cluster, cosmology, mantz_data_dir
    # use the MCMC files produced for the pressure profile project
    redshift = params_dict['redshift']
    cluster = params_dict['mantz_cluster_name']
    cosmology = params_dict['mantz_cosmology']
    data_dir = params_dict['mantz_data_dir']

    r2500 = params_dict['r2500']

    # get the mass and scale radius
    # r500_m500_path = os.path.join(data_dir, 'Mantz_Xray_masses',
    #                             cosmology, cluster + '_r500_M500.txt')
    # this_xray_r500, this_xray_m500 = np.split(np.loadtxt(r500_m500_path).T, 2, axis=0)

    # r500 = np.median(this_xray_r500)
    # r2500 = 0.71
    # m500 = np.median(this_xray_m500) * 1.e-14

    # get the pressure profile values
    conversions_path = os.path.join(data_dir, 'Mantz_Xray_pressures',
                                    cosmology, cluster + '_conversions.txt')
    xray_conversions = []
    with open(conversions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' ').split(' ')
            if line[0][0] != '-' and len(line) == 3:
                xray_conversions.append(float(line[1]))
    xray_conversions = np.array(xray_conversions)

    pressure_path = os.path.join(data_dir, 'Mantz_Xray_pressures',
                                 cluster + '_pressure')
    pressure_dat_path = pressure_path + '.dat'
    pressure_float_data = np.loadtxt(pressure_dat_path, skiprows=1, usecols=(0, 4, 5, 6)).T
    xray_radius, xray_pressure, xray_pressure_lo, xray_pressure_hi = np.split(pressure_float_data, 4, axis=0)
    xray_radius = xray_radius[0]    # get 1d np array
    pressure_txt_path = pressure_path + '.txt'
    xray_pressure = np.loadtxt(pressure_txt_path).T
    xray_use = np.loadtxt(pressure_dat_path, skiprows=1, usecols=(3), dtype=str).T
    
    valid_xray = np.nonzero((xray_use == 'TRUE'))[0]
    xray_pressure = xray_pressure[valid_xray]
    # xray_radius = xray_radius[valid_xray] * xray_conversions[1] / r500[0]
    xray_radius = xray_radius[valid_xray] * xray_conversions[1] / r2500

    # get the temperature profile values
    temperature_path = os.path.join(data_dir, 'Mantz_Xray_temperatures',
                                    cosmology, cluster + '_temperature.txt')
    xray_t = np.loadtxt(temperature_path).T
    xray_temperature = xray_t[valid_xray]

    # create a data cube to integrate
    dx = 0.01
    max_x = 1
    max_z = np.max(xray_radius)
    x1d = np.arange(np.round(max_x*2/dx)+1) * dx
    x1d = x1d - np.mean(x1d)
    z1d = np.arange(np.round(max_z*2/dx)+1) * dx
    z1d = z1d - np.mean(z1d)
    nx = np.size(x1d)
    nz = np.size(z1d)
    x3d = np.zeros((nx, nx, nz))
    y3d = np.zeros((nx, nx, nz))
    z3d = np.zeros((nx, nx, nz))
    for iy in range(nx):
        for iz in range(nz):
            x3d[:, iy, iz] = x1d
    for ix in range(nx):
        for iz in range(nz):
            y3d[ix, :, iz] = x1d
    for ix in range(nx):
        for iy in range(nx):
            x3d[ix, iy, :] = z1d
    r3d = np.sqrt(x3d**2 + y3d**2 + z3d**2)
    r2d = np.sqrt(x3d**2 + y3d**2)
    valid = (r2d <= 1)  # not actually translation from IDL, but easier for Python

    un_normalized_density = xray_pressure / xray_temperature
    n_jk = 100
    mw_temperature = np.zeros((n_jk))
    pw_temperature = np.zeros((n_jk))
    for i_jk in range(n_jk):
        temperature_3d = np.interp(r3d, xray_radius, xray_temperature[:, i_jk])
        density_3d = np.interp(r3d, xray_radius, un_normalized_density[:, i_jk])
        pressure_3d = np.interp(r3d, xray_radius, xray_pressure[:, i_jk])

        mw_temperature[i_jk] = np.sum(temperature_3d[valid] * density_3d[valid]) / np.sum(density_3d[valid])
        pw_temperature[i_jk] = np.sum(temperature_3d[valid] * pressure_3d[valid]) / np.sum(pressure_3d[valid])

    # correct for bias of 0.09 +- 0.13 found in Wan et al.
    T_mw = np.median(mw_temperature) / 1.09
    print(f"T_mw = {T_mw}")
    sigma_T_mw = np.sqrt((np.std(mw_temperature) / 1.09)**2 + (np.median(mw_temperature) / 1.09 * 0.13)**2)
    print(f"sigma_T = {sigma_T_mw}")

    # correct for bias of 0.09 +- 0.13 found in Wan et al.
    T_pw = np.median(pw_temperature) / 1.09
    print(f"T_pw = {T_pw}")
    sigma_T_pw = np.sqrt((np.std(pw_temperature) / 1.09)**2 + (np.median(pw_temperature) / 1.09 * 0.13)**2)
    print(f"sigma_T = {sigma_T_pw}")

    return T_mw, sigma_T_mw, T_pw, sigma_T_pw

# translated from Jack's IDL code with modifications
def compute_sz_spectrum(nu, optical_depth=0.01, temperature=10., vpec=0., fast=False, cmb_units=False, file_prefix=''):

    if vpec < 0:
        vpec = np.abs(vpec)
        vpec_direction = -1.0
    else:
        vpec_direction = 1.0

    # use a reasonable number of points in the calculation
    npoints = 100

    # find the min/max frequencies
    kB = 1.381e-23
    hPlanck = 6.626e-34
    Tcmb = 2.725
    cLight = 3.e8
    x = nu * hPlanck / (kB * Tcmb)
    min_x = min(x) * 0.8
    max_x = max(x) / 0.8

    # call Python SZpack bindings
    xo =  np.geomspace(min_x, max_x, num=npoints)

    betao = 0.001241
    muo = 0

    Te_order = 20           # ignored in 3D, 5D
    betac_order = 2         # ignored in 3D, 5D
    eps_Int = 1.0e-4        # only used in 3D, 5D

    kmax = 4                # only for CNSNopt
    accuracy_level = 2      # only for CNSNopt

    if fast:
        sz_output = szpack_bindings.output_SZ_distortion_asymptotic_vector(xo, optical_depth, temperature, vpec, vpec_direction, betao, muo, Te_order, betac_order)
    else:
        sz_output = szpack_bindings.output_SZ_distortion_5D_vector(xo, optical_depth, temperature, vpec, vpec_direction, betao, muo, eps_Int)

    # extract the output of SZpack
    xarr, val1, val2 = zip(*sz_output)

    # interpolate to the desired frequencies
    spectrum = np.interp(x[:nu.size], xarr, val1)
    
    # multiply by I0 in order to obtain output in MJy/ster
    I0 = 2. * ( (kB * Tcmb) / (hPlanck * cLight)**(2./3.) )**3. * 1.e20
    spectrum = spectrum * I0

    if cmb_units:
        raise Exception('cmb_units flag Not implemented')
        # x = hPlanck * nu / (kB * Tcmb)
        # h = x * np.exp(x) / (np.exp(x) - 1.)
        # ICMB = planck_bnu(nu, Tcmb) / 1.e-20
        # spectrum = Tcmb * 1.e6 / h * spectrum / ICMB

    return spectrum

'''
def calc_spire_band_centers():
    t_array = np.arange(151.)/2. + 1.e-7
    
    # in place of IDL's printf
    outfile = 'data/rSZ/band_centers_20210810.txt'
    f_buffer = []

    bolocam = np.full((t_array.size), 0.)
    plw = np.full((t_array.size), 0.)
    pmw = np.full((t_array.size), 0.)
    psw = np.full((t_array.size), 0.)

    def get_freq_trans(spire_dat_fpath):
            wavelength, trans = np.split(np.loadtxt(spire_dat_fpath).T, 2, axis=0)
            # get 1d np array
            wavelength = wavelength[0]
            trans = trans[0]

            freq = 3.e8 / wavelength * 1.e10
            return freq, trans

    freq_plw, trans_plw = get_freq_trans('data/rSZ/Herschel_SPIRE.PLW.dat')
    freq_pmw, trans_pmw = get_freq_trans('data/rSZ/Herschel_SPIRE.PMW.dat')
    freq_psw, trans_psw = get_freq_trans('data/rSZ/Herschel_SPIRE.PSW.dat')

    
    for i_temp in range(2):#range(t_array.size):
        temperature = t_array[i_temp]
        print(temperature)

        SZ_sig = compute_sz_spectrum(freq_plw, temperature=temperature)
        plw[i_temp] = np.sum(freq_plw * SZ_sig * trans_plw) / np.sum(SZ_sig * trans_plw) * 1.e-9

        SZ_sig = compute_sz_spectrum(freq_pmw, temperature=temperature)
        pmw[i_temp] = np.sum(freq_pmw * SZ_sig * trans_pmw) / np.sum(SZ_sig * trans_pmw) * 1.e-9
        
        SZ_sig = compute_sz_spectrum(freq_psw, temperature=temperature)
        psw[i_temp] = np.sum(freq_psw * SZ_sig * trans_psw) / np.sum(SZ_sig * trans_psw) * 1.e-9

        # super duper big extrapolation here
        # restore IDL .sav
        trans_mm_sav = sp.io.readsav('data/idl_savs/trans_1.5mm.sav')
        # atm_trans_interp = np.interp(freq_psw, trans_mm_sav['nu'], trans_mm_sav['trans_2mm'])
        interp = sp.interpolate.interp1d(trans_mm_sav['nu'], trans_mm_sav['trans_2mm'], fill_value="extrapolate")
        atm_trans_interp = interp(freq_psw)
        trans = trans_psw * atm_trans_interp
        
        spectra_mm_sav = sp.io.readsav('data/idl_savs/2mm_spectra.sav')
        spec_nu = spectra_mm_sav['spec_nu']
        spec = spectra_mm_sav['spec']

        lissajous_sav = sp.io.readsav('data/idl_savs/coadd_clean_lissajous_skysub_250mHz_psdfit_nosig.sav')
        mapstruct = lissajous_sav['mapstruct']
        freq = spec_nu / 1.e9
        trans = spec[:, mapstruct['goodbolos'][0]]
        trans = np.median(trans, axis=1)
        interp = sp.interpolate.interp1d(trans_mm_sav['nu'], trans_mm_sav['trans_15mm'], fill_value="extrapolate")
        atm_trans_interp = interp(freq)
        trans = trans * atm_trans_interp
        trans[(freq >= 170.) | (freq <= 120.)] = 0
        SZ_spec = compute_sz_spectrum(freq*1.e9, temperature=temperature)
        
        bolocam[i_temp] = np.sum(freq * SZ_spec * trans) / np.sum(SZ_spec * trans)
        f_buffer.append([temperature, bolocam[i_temp], plw[i_temp], pmw[i_temp], psw[i_temp]])
    
    np.savetxt(outfile, f_buffer, fmt='%.2f', header='T (keV)\tBolocam\tPLW\tPMW\tPSW', delimiter='\t')
'''

# translated from Jack's IDL code with modifications
# params_dict must have keys 'trans_1.5mm', '2mm_spectra', 'coadd_clean_lissajous_skysub_250mHz_psdfit_nosig'
def calc_bolo_and_spire_band_centers(T_pw, params_dict):
    # T_pw is pressure_weighted X-ray temperature

    def get_freq_trans(spire_dat_fpath):
            wavelength, trans = np.split(np.loadtxt(spire_dat_fpath).T, 2, axis=0)
            # get 1d np array
            wavelength = wavelength[0]
            trans = trans[0]

            freq = 3.e8 / wavelength * 1.e10
            return freq, trans

    freq_plw, trans_plw = get_freq_trans('data/rSZ/Herschel_SPIRE.PLW.dat')
    freq_pmw, trans_pmw = get_freq_trans('data/rSZ/Herschel_SPIRE.PMW.dat')
    freq_psw, trans_psw = get_freq_trans('data/rSZ/Herschel_SPIRE.PSW.dat')

    SZ_sig = compute_sz_spectrum(freq_plw, temperature=T_pw)
    plw = np.sum(freq_plw * SZ_sig * trans_plw) / np.sum(SZ_sig * trans_plw) * 1.e-9

    SZ_sig = compute_sz_spectrum(freq_pmw, temperature=T_pw)
    pmw = np.sum(freq_pmw * SZ_sig * trans_pmw) / np.sum(SZ_sig * trans_pmw) * 1.e-9
    
    SZ_sig = compute_sz_spectrum(freq_psw, temperature=T_pw)
    psw = np.sum(freq_psw * SZ_sig * trans_psw) / np.sum(SZ_sig * trans_psw) * 1.e-9

    # super duper big extrapolation here
    # restore IDL .sav
    trans_mm_sav = sp.io.readsav(params_dict['trans_1.5mm'])
    # atm_trans_interp = np.interp(freq_psw, trans_mm_sav['nu'], trans_mm_sav['trans_2mm'])
    # interp = sp.interpolate.interp1d(trans_mm_sav['nu'], trans_mm_sav['trans_2mm'], fill_value="extrapolate")
    # atm_trans_interp = interp(freq_psw)
    # trans = trans_psw * atm_trans_interp
    
    spectra_mm_sav = sp.io.readsav(params_dict['2mm_spectra'])
    spec_nu = spectra_mm_sav['spec_nu']
    spec = spectra_mm_sav['spec']

    lissajous_sav = sp.io.readsav(params_dict['coadd_clean_lissajous_skysub_250mHz_psdfit_nosig'])
    mapstruct = lissajous_sav['mapstruct']
    freq = spec_nu / 1.e9
    trans = spec[:, mapstruct['goodbolos'][0]]
    trans = np.median(trans, axis=1)
    interp = sp.interpolate.interp1d(trans_mm_sav['nu'], trans_mm_sav['trans_15mm'], fill_value="extrapolate")
    atm_trans_interp = interp(freq)
    trans = trans * atm_trans_interp
    trans[(freq >= 170.) | (freq <= 120.)] = 0
    SZ_spec = compute_sz_spectrum(freq*1.e9, temperature=T_pw)
    
    bolocam = np.sum(freq * SZ_spec * trans) / np.sum(SZ_spec * trans)
    return bolocam, plw, pmw, psw

# params_dict must have keys 'bandpasses_act', 'full_ivar_90', 'full_ivar_150'
def get_act_band_centers(T_pw, params_dict):
    bandpasses_txt = np.loadtxt(params_dict['bandpasses_act'])
    bandpasses_by_freq = {freq: bandpasses for freq, bandpasses in zip(bandpasses_txt[:, 0], bandpasses_txt[:, 1:])}
    def get_act_band_center(full_ivar_path, T_pw):
        noisebox = enmap.read_map(full_ivar_path)
        weights = np.sum(noisebox, axis=(0, 2, 3, 4)) / np.sum(noisebox)
        bandpass_by_freq = {freq: np.sum(bandpasses * weights) for freq, bandpasses in bandpasses_by_freq.items()}
        nus = np.array(list(bandpass_by_freq.keys()))
        Bs = np.array(list(bandpass_by_freq.values()))
        SZs = compute_sz_spectrum(nus, temperature=T_pw)

        trans_mm_sav = sp.io.readsav('data/idl_savs/trans_1.5mm.sav')
        interp = sp.interpolate.interp1d(trans_mm_sav['nu'], trans_mm_sav['trans_15mm'], fill_value="extrapolate")
        atm_trans_interp = interp(nus)
        Bs = Bs * atm_trans_interp

        band_center = np.sum(nus * Bs * SZs) / np.sum(Bs * SZs)
        return band_center
        
    band_center_90 = get_act_band_center(params_dict['full_ivar_90'], T_pw)
    band_center_150 = get_act_band_center(params_dict['full_ivar_150'], T_pw)
    return band_center_90, band_center_150


def fit_tau_e(nu_bolocam, nu_90, nu_150, dI_bolocam, dI_90, dI_150, T_pw):
    # solve for tau_e
    # * 100 for 0.01 optical depth
    SZ_bolocam = compute_sz_spectrum(np.array([nu_bolocam * 1.e9]), temperature=T_pw)[0] * 100
    SZ_90 = compute_sz_spectrum(np.array([nu_90 * 1.e9]), temperature=T_pw)[0] * 100
    SZ_150 = compute_sz_spectrum(np.array([nu_150 * 1.e9]), temperature=T_pw)[0] * 100
    a = np.array([[SZ_bolocam, SZ_90, SZ_150]]).T
    b = np.array([[dI_bolocam, dI_90, dI_150]]).T

    return np.linalg.lstsq(a, b, rcond=None)[0].item()   
