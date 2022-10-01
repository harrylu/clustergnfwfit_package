import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import scipy as sp
import numpy as np

def apply_gaussian_blur(sigma_x, sigma_y, map):
    # map is input array

    # Apply gaussian filter
    sigma = [sigma_y, sigma_x]
    y = sp.ndimage.gaussian_filter(map, sigma, mode='constant')
    return y

def imshow_gaussian_blur_default(sigma_x, sigma_y, map, vmin=-100, vmax=100):
    """Imshows the plot with extra parameters

    Args:
        sigma_x (float): 
        sigma_y (float): 
        map (2d array):
        vmin (float): values <= vmin will be darkest blue
        vmax (float): values >= vmax will be darkest red
    """

    plt.imshow(apply_gaussian_blur(sigma_x, sigma_y, map), cmap=cm.coolwarm, interpolation='nearest', vmin=vmin,
               vmax=vmax)

def radial_profile(data, x_offset=0, y_offset=0, pixels_per_bin=2):
    """Calculates radial mean vs radius of 2d array. Assumes center is at center of 2d array data.

    Args:
        data (2d array): returns radial profile of this
        x_offset (float, optional): offset from center in pixels -
        Positive is rightward. Defaults to 0.
        y_offset (float, optional): offset from center in pixels -
        Positive is downward. Defaults to 0.
        pixels_per_bin (int, optional): depth of bin in pixels.
        Bins extend radially outward. Defaults to 2.

    Returns:
        1d array: contains radial means of bins with closest bins first
    """

    # positive offsets are right and down
    center = ((data.shape[1] - 1) / 2 + y_offset, (data.shape[0] - 1) / 2 + x_offset)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2) / pixels_per_bin   # 1 pixel r for 30 arcseconds, 2 pixel r for 60 arcseconds
    r = r.astype(int)   # r rounds down

    # add data to respective r bins (from 0 to r_max); (aka adds rs to r bins, weight of data value)
    tbin = np.bincount(r.ravel(), data.ravel())
    # get number of data points in each bin (aka add rs to r bins, weight of 1)
    nr = np.bincount(r.ravel())
    # radial mean at r is sum of data in r bin / number of data points in r bin
    radialprofile = tbin / nr
    return radialprofile

def plot_polar_map_radial_profile(r_profile):
    """Assigns a polar map of the radial profile to the current matplotlib figure.

    Args:
        r_profile (_type_): radial profile: radial means with closest first

    Returns:
    None
    """
    rs = [r for r in range(len(r_profile))]
    ax = plt.subplot(projection='polar')
    norm = colors.Normalize(vmin=-100, vmax=100)
    for r, mean in zip(rs, r_profile):
        ax.bar(0, 1, 2 * np.pi, r, color=cm.coolwarm(norm(mean)))
    # get rid of degree markings
    ax.set_xticklabels([])
    plt.gcf().colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)