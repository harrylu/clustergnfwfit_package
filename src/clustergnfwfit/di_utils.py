import numpy as np
from .conversions import convert_microkelvin_to_mjysr


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


def calc_sigma_di(freq, sigma_P0, P0, di):
    """Calculates the one sigma error for di (not actually correct way)

    Args:
        freq (float): in GHz
        sigma_P0 (float): one sigma error for P0
        P0 (float):
        di (float): di of cluster

    Returns:
        float: approximate one sigma error of cluster
    """
    sigma_P0 = convert_microkelvin_to_mjysr(sigma_P0, freq)
    P0 = convert_microkelvin_to_mjysr(P0, freq)
    
    return sigma_P0 * di / P0
    