import numpy as np
from conversions import convert_microkelvin_to_mjysr


def get_R2500_avg(map, arcseconds_per_pixel, R2500):
    """AKA get di value.

    Args:
        map (2d array): gNFW is centered in this array
        arcseconds_per_pixel (int): length of pixel in arcseconds
        R2500 (float): cluster R2500 in arcseconds.
        Pixels within R2500 will be used in calculation.

    Returns:
        float: Returns the average of all pixels in map that are within R2500.
    
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

    
if __name__ == "__main__":
    import ellipsoid_model
    import emcee
    import matplotlib.pyplot as plt
    # params, perror = gnfw_fit_map,gnfw_fit_map(...)

    # I will set the params manually for this demo
    # based on previous fit done on MACSJ0025.4

    # This part demonstrates making the map that will go into the fits file.
    # 470*470 pixels with 4 arcsecond pixels

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


    first_fit_backend = 'emcee_backend_7777.h5'
    first_reader = emcee.backends.HDFBackend(first_fit_backend)

    second_fit_backend = 'emcee_backend_2nd_7777.h5'
    second_reader = emcee.backends.HDFBackend(second_fit_backend)

    # burnin
    first_burnin=2000
    second_burnin=2000
    # samples is shape (nsamples, nwalkers, nparameters)
    first_samples = first_reader.get_chain(discard=first_burnin)
    second_samples = second_reader.get_chain(discard=second_burnin)

    # reshape -> (ntotalsamples, nparameters)
    second_samples = second_samples.reshape((-1, second_samples.shape[-1]))
    samples_cbrt_p0_90 = second_samples[:, 0]
    samples_cbrt_p0_150 = second_samples[:, 1]
    samples_cbrt_p0_bolocam = second_samples[:, 2]

    first_medians = np.median(first_samples, axis=(0, 1))
    theta, _, _, r_x, r_y, offset_x, offset_y, _, _, _, _ = first_medians

    r_z = np.sqrt(r_x*r_y)

    samples_p0_90_mjysr = convert_microkelvin_to_mjysr(samples_cbrt_p0_90 ** 3, 90)
    samples_p0_150_mjysr = convert_microkelvin_to_mjysr(samples_cbrt_p0_150 ** 3, 150)
    samples_p0_bolocam_mjysr = convert_microkelvin_to_mjysr(samples_cbrt_p0_bolocam ** 3, 140)
    
    P0_90, P0_150, P0_bolocam = np.median(samples_p0_90_mjysr), np.median(samples_p0_150_mjysr), np.median(samples_p0_bolocam_mjysr)
    P0_90_std, P0_150_std, P0_bolocam_std = np.std(samples_p0_90_mjysr), np.std(samples_p0_150_mjysr), np.std(samples_p0_bolocam_mjysr)
    

    model_no_c = ellipsoid_model.eval_pixel_centers(theta, 1, r_x, r_y, r_z, 4, R500, offset_x=offset_x, offset_y=offset_y, img_height=470, img_width=470)
    model_90 = P0_90 * model_no_c
    model_150 = P0_150 * model_no_c
    model_bolocam = P0_bolocam * model_no_c

    di_model_no_c = get_R2500_avg(model_no_c, 4, R2500)
    dis_90 = convert_microkelvin_to_mjysr(di_model_no_c * (samples_cbrt_p0_90 ** 3), 90)
    dis_150 = convert_microkelvin_to_mjysr(di_model_no_c * (samples_cbrt_p0_150 ** 3), 150)
    dis_bolocam = convert_microkelvin_to_mjysr(di_model_no_c * (samples_cbrt_p0_bolocam ** 3), 140)

    # now, get di values
    # di_90 = get_R2500_avg(model_90, 4, R2500)
    # di_150 = get_R2500_avg(model_150, 4, R2500)
    # di_bolocam = get_R2500_avg(model_bolocam, 4, R2500)
    di_90 = np.median(dis_90)
    di_150 = np.median(dis_150)
    di_bolocam = np.median(dis_bolocam)

    sigma_90 = np.std(dis_90)
    sigma_150 = np.std(dis_150)
    sigma_bolocam = np.std(dis_bolocam)


    print(f'di 150: {di_150}')
    print(f'di 90: {di_90}')
    print(f'di bolocam: {di_bolocam}')

    print('sigma di 150', sigma_150)
    print('sigma di 90', sigma_90)
    print('sigma di bolocam', sigma_bolocam)

    # plt.figure('model 150')
    # plt.imshow(model_150)

    # plt.figure('model 90')
    # plt.imshow(model_90)

    # plt.figure('model bolo')
    # plt.imshow(model_bolocam)

    plt.show()