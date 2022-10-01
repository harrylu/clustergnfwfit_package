"""
Contains functions relevant to computing a map of a gNFW model
as well as the di value of a map (average with R2500).
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
from multiprocessing import Pool

def _f_gnfw(z, x, y, p0, rs, alpha=1.05, beta=5.49, gamma=0.31):
    """Evaluates gNFW

    Args:
        z (float): arcseconds
        x (float): arcseconds
        y (float): arcseconds
        p0 (float): 
        rs (float): arcseconds
        alpha (float, optional): . Defaults to 1.05.
        beta (float, optional): . Defaults to 5.49.
        gamma (float, optional): . Defaults to 0.31.

    Returns:
        float: output of gNFW
    """
    r = np.sqrt(z ** 2 + x ** 2 + y ** 2)
    r_ratio = r / rs
    return p0 / ((r_ratio) ** gamma * (1 + r_ratio ** alpha) ** ((beta - alpha) / gamma))

def _rep_gnfw2d(num_samples, P0, RS, R500):
    """Approximates the value of gnfw2d(r) (which gives the dV behind that point) on the 2d map with r in arcseconds

    Args:
        num_samples (int): number of samples to evaluate.
        Samples are x coordinates log spaced in interval (0.001, 5*R500)
        P0 (float): P0 of cluster
        RS (float): RS of cluster - arcseconds
        R500 (float): R500 of cluster (all values outside 5*R500 eval to 0)

        Returns:
            function: call method uses interpolation to find
            the value of gnfw2d(r).
    """

    # we are integrating along axis z with x = r (because y = 0) at logspaced steps
    xs = np.geomspace(0.001, 5 * R500, num=num_samples)
    integrated_gnfw_of_r = []
    for x in xs:
        # argument 3 of scipy.integrate.quad is the upper bound:
        # we only want to integrate within 5*R500 so we solve
        # x^2 + y^2 + z^2 = (5*R500)^2, y is 0
        # so z = sqrt((5*R500)^2 - x^2)

        # multiply integration by 2 for the other z direction
        integrated_gnfw_of_r.append(
            scipy.integrate.quad(_f_gnfw, 0, np.sqrt((5 * R500) ** 2 - x ** 2), args=(x, 0, P0, RS))[0] * 2
            )
    # fill value is 0 because gnfw(r>5*R500)=0
    interp = scipy.interpolate.interp1d(xs, integrated_gnfw_of_r, kind='linear', bounds_error=False, fill_value=0)
    return interp


def _eval_pixel(pixel_x, pixel_y, center_pix_x, center_pix_y, arcseconds_per_pixel, interp, R500, epsrel):
    """For the specified pixel on the 2d gNFW map, get its value
    using double integral over x and y.

    Args:
        pixel_x (int): pixel's x index
        pixel_y (int): pixel's y index
        center_pix_x (int): gNFW map's center pixel's x index
        center_pix_y (int): gNFW map's center pixel's y index
        arcseconds_per_pixel (int): length of pixel in arcseconds
        interp (function): call method must return the value of
        gnfw(r) on the 2d map
        R500 (float): R500 of cluster in arcseconds (all values outside 5*R500 eval to 0)
        epsrel (float): epsrel for scipy dblquad function (affects speed and accuracy)

    Returns:
        tuple: first element is (pixel_x, pixel_y)
        second element is return of dblquad:
        (float - The value of the pixel,
        float - An estimate of the error.)
    """

    # need to write f(x, y) that returns dV at pixel distance (dist_x, dist_y) for integration
    def f(dist_x, dist_y):
        # convert pixel distance to arcsecond distance
        r = np.sqrt(dist_x ** 2 + dist_y ** 2) * arcseconds_per_pixel
        # print(r, interp(r))
        if r == 0:
            return interp(0.001)  # don't want to return infinity
        return interp(r)

    # pixel distances of this pixel from center pixel
    dist_x = np.abs((pixel_x - center_pix_x))
    dist_y = np.abs((pixel_y - center_pix_y))

    r = np.sqrt(dist_x ** 2 + dist_y ** 2) * arcseconds_per_pixel
    if r > 5*R500 + arcseconds_per_pixel:
        return (pixel_x, pixel_y), (0,0)

    # integrate to find volume under pixel surface
    # pixels are centered at the (idx_x, idx_y) so need to add/subtract 0.5 to get the pixel bounds
    # we should have to divide by pixel area after integration but in this case, the pixel area is 1
    return (pixel_x, pixel_y), scipy.integrate.dblquad(f, dist_x - 0.5, dist_x + 0.5, dist_y - 0.5, dist_y + 0.5,
                                                    epsabs=0, epsrel=epsrel)

eval_pixel_multi_results = []
def eval_pixel_multi(args_list, num_processes):
    """Wrapper around _eval_pixel for multiprocessing. Writes results to eval_pixel_multi_results.

    Args:
        args_list (list): Elements are iterables containing arguments to pass into _eval_pixel.
        num_processes (int): Max number of cores to use

    Returns:
        Returns None after all the pixels are done being evaluated.
        Evaluated results are in the list eval_pixel_multi_results.
    """
    
    eval_pixel_multi_results.clear()
    pool = Pool(processes=num_processes)
    res = [pool.apply_async(_eval_pixel, args=args, callback=_eval_pixel_multi_callback) for args in args_list]
    for a in res:
        a.wait()

def _eval_pixel_multi_callback(result):
    eval_pixel_multi_results.append(result)

def make_los_gnfw_grid(P0, RS, R500, height, width, offset_x, offset_y, arcseconds_per_pixel, epsrel, num_processes):
    """Numerically integrates the spherically symmetric gNFW function along
    line of sight onto a 2d plane.

    Args:
        P0 (float): P0 of model
        RS (float): RS of model - arcseconds
        R500 (float): R500 (arcseconds) of the cluster being analyzed.
        All values outside of 5*R500 will be ignored in the integration.
        height (int): map height in pixels
        width (int): map width in pixels
        offset_x (float): offset (in pixels) of gNFW center from the center of
        the returned map (positive to the right)
        offset_y (float): offset (in pixels) of gNFW center from the center of
        the returned map (positive is down)
        arcseconds_per_pixel (int): length of pixel in arcseconds
        epsrel (float): epsrel for scipy dblquad function (affects speed and accuracy)
        num_processes (int): Max number of cores to use

    Returns:
        2d array: line-of-sight integrated gNFW model.
    """

    # gNFW(r) in 2d space
    interp = _rep_gnfw2d(100, P0, RS, R500)

    # grid creation below:
    # positive offsets are right and down
    center_pix_x = (width - 1) / 2 + offset_x
    center_pix_y = (height - 1) / 2 + offset_y
    grid = np.empty((height, width))

    # regular (not multiprocess) approach
    #for pixel_x in range(grid.shape[1]):
    #    for pixel_y in range(grid.shape[0]):
    #        pixel_pos, quad_return = _eval_pixel(pixel_x, pixel_y, center_pix_x, center_pix_y,
    #                                    arcseconds_per_pixel, interp, R500, epsrel)
    #        pix_x, pix_y = pixel_pos
    #        y, _ = quad_return
    #        grid[pix_y, pix_x] = y

    # multiprocess
    args_list = []
    for pixel_y in range(grid.shape[0]):
        for pixel_x in range(grid.shape[1]):
            args_list.append((pixel_x, pixel_y, center_pix_x, center_pix_y,
                                        arcseconds_per_pixel, interp, R500, epsrel))
    
    eval_pixel_multi(args_list, num_processes=num_processes)
    for result in eval_pixel_multi_results:
        pixel_pos, quad_return = result
        pix_x, pix_y = pixel_pos
        y, _ = quad_return
        grid[pix_y, pix_x] = y
    eval_pixel_multi_results.clear()

    return grid

def make_fits_grid(P0, RS, R500, arcseconds_per_pixel, epsrel, num_processes):
    """Basically the same as make_los_gnfw_grid but with some optimizations.
    Compared to make_los_gnfw_grid,there are no x/y offsets and the grid is
    fixed 470*470 pixels.

    Args:
        P0 (float): P0 of model
        RS (float): RS of model (arcseconds)
        R500 (float): R500 (arcseconds)
        arcseconds_per_pixel (int): length of pixel in arcseconds
        epsrel (float): epsrel for scipy dblquad function (affects speed and accuracy)
        num_processes (int): Max number of cores to use

    Returns:
        2d array: line-of-sight integrated gNFW model.
    """

    width = 470
    height = 470

    # gNFW(r) in 2d space
    interp = _rep_gnfw2d(100, P0, RS, R500)

    # grid creation below:
    # positive offsets are right and down
    # 0.5 in this special case cuz grid is even
    center_pix_x = -0.5
    center_pix_y = -0.5

    grid = np.empty((height//2, width//2))

    # we will make bottom right quadrant (actually only need part of the quadrant)
    # split quadrant into L shapes, evaluate vertical part of the L, fill in the rest of the L with the vertical values rotated cw 90 degrees
    args_list = []
    for pixel_x in range(grid.shape[1]):
        # calculate for each pixel in the column (vertical part of the L shape)
        for pixel_y in range(pixel_x, grid.shape[0]):
            args_list.append((pixel_x, pixel_y, center_pix_x, center_pix_y,
                                        arcseconds_per_pixel, interp, R500, epsrel))

    eval_pixel_multi(args_list, num_processes=num_processes)
    for result in eval_pixel_multi_results:
        pixel_pos, quad_return = result
        pix_x, pix_y = pixel_pos
        y, _ = quad_return
        grid[pix_y, pix_x] = y
        # fill in horizontal part of the L
        grid[pix_x, pix_y] = y
    eval_pixel_multi_results.clear()

    # reflect left horizontally to get bottom left quadrant
    grid = np.pad(grid, [(0, 0), (grid.shape[1], 0)], 'symmetric')

    # reflect up vertically to get top two quadrants
    grid = np.pad(grid, [(grid.shape[0], 0), (0, 0)], 'symmetric')

    return grid
