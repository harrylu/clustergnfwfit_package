import os
import numpy as np

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
    xray_radius = xray_radius[0]
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
    weight = np.outer(xray_radius**2 * un_normalized_density[:, 0], np.full((n_jk), 1))
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

def calc_spire_band_centers():
    from pixell import enmap
    import os
    MAP_FITS_DIR = "/home/harry/clustergnfwfit_package/data/map_fits_files"
    FNAME_FULLIVAR_90 = "act_planck_dr5.01_s08s18_AA_f090_night_fullivar.fits"
    FNAME_FULLIVAR_150 = "act_planck_dr5.01_s08s18_AA_f150_night_fullivar.fits"

    auxillary_dir = '/home/harry/clustergnfwfit_package/data/act_dr5.01_auxilliary'
    fname_bandpasses_txt = 'act_planck_dr5.01_s08s18_bandpasses.txt'
    path_bandpasses_txt = os.path.join(auxillary_dir, fname_bandpasses_txt)

    path_90 = os.path.join(MAP_FITS_DIR, FNAME_FULLIVAR_90)
    path_150 = os.path.join(MAP_FITS_DIR, FNAME_FULLIVAR_150)

    bandpasses_txt = np.loadtxt(path_bandpasses_txt)
    bandpasses_by_freq = {freq: bandpasses for freq, bandpasses in zip(bandpasses_txt[:, 0], bandpasses_txt[:, 1:])}
    def get_act_band_centers(full_ivar_path):
        noisebox = enmap.read_map(full_ivar_path)
        weights = np.sum(noisebox, axis=(0, 2, 3, 4)) / np.sum(noisebox)
        bandpass_by_freq = {freq: np.sum(bandpasses * weights) for freq, bandpasses in bandpasses_by_freq.items()}


    noisebox_90 = enmap.read_map(path_90)
    weights_90 = np.sum(noisebox_90, axis=(0, 2, 3, 4)) / np.sum(noisebox_90)
    bandpass_by_freq_90 = {freq: np.sum(bandpasses * weights_90) for freq, bandpasses in bandpasses_by_freq.items()}

    noisebox_150 = enmap.read_map(path_150)
    weights_150 = np.sum(noisebox_150, axis=(0, 2, 3, 4)) / np.sum(noisebox_150)
    bandpass_by_freq_150 = {freq: np.sum(bandpasses * weights_150) for freq, bandpasses in bandpasses_by_freq.items()}
    

if __name__ == "__main__":
    calc_spire_band_centers()