import numpy as np
import scipy
import scipy.integrate
import math
import cProfile

# Ellipsoid defined by three axes, axes along x and y are free
# also rotation in x-y plane

def f_gnfw_ellipsoid(x, y, z, p0, r_x, r_y, r_z, R500, alpha=1.05, beta=5.49, gamma=0.31):
    """Evaluates ellipsoid gNFW

    Args:
        z (float): arcseconds
        x (float): arcseconds
        y (float): arcseconds
        p0 (float): 
        r_x (float): length of major axis (arcseconds)
        r_y (float): length of minor axis (arcseconds)
        r_z (float): length of semi-major axis (arcseconds)
        alpha (float, optional): . Defaults to 1.05.
        beta (float, optional): . Defaults to 5.49.
        gamma (float, optional): . Defaults to 0.31.

    Returns:
        float: output of gNFW
    """
    s = np.sqrt((x/r_x) ** 2 + (y/r_y) ** 2 + (z/r_z) ** 2)
    return f_gnfw_s_ellipsoid(s, p0, alpha, beta, gamma)

def f_gnfw_s_ellipsoid(s, p0, alpha=1.05, beta=5.49, gamma=0.31):
    return p0 / ((s) ** gamma * (1 + s ** alpha) ** ((beta - alpha) / gamma))

# s_xy_sqr = s_xy^2 = (x/r_x)^2 + (y/r_y)^2
def f_gnfw_s_xy_sqr_ellipsoid(s_xy_sqr, z, r_z, p0, alpha=1.05, beta=5.49, gamma=0.31):
    s = np.sqrt(s_xy_sqr + (z/r_z)**2)
    return f_gnfw_s_ellipsoid(s, p0, alpha, beta, gamma)

# theta in degrees
def get_2d_rot_mat(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def f_gnfw_ellipsoid_with_rot(theta, x, y, z, *args):
    """Evaluates ellipsoid gNFW that is rotated theta degrees in x, y plane

    Args:
        theta (float): degrees
        z (float): arcseconds
        x (float): arcseconds
        y (float): arcseconds

    Returns:
        float: output of gNFW
    """
    # theta is negative because we are rotating the point instead of the model
    R = get_2d_rot_mat(-theta)
    xy = np.array([[x], [y]])
    rotated = R @ xy
    return f_gnfw_ellipsoid(rotated[0].item(), rotated[1].item(), z, *args)

# ONLY FOR TESTING PURPOSES; TOO SLOW OTHERWISE
# pixel_coords_to_eval is list of tuples (x,y)
def full_solution(pixels_coords_to_eval, theta, p0, r_x, r_y, r_z, arcseconds_per_pixel, R500, offset_x=0, offset_y=0, img_width=470, img_height=470, epsabs=1.49e-8, epsrel=1.49e-8):
    # theta is ccw rotation in x,y plane
    # let's try no offset first
    # x^2 + y^2 + z^2 = (5*R500)^2, y is 0
    # so z = +-sqrt((5*R500)^2 - x^2 - y^2) <= these are bounds
    center_pix_x = (img_width - 1) / 2 + offset_x
    center_pix_y = (img_height - 1) / 2 + offset_y
    pixels = np.zeros((img_width, img_height))
    for pix_x, pix_y in pixels_coords_to_eval:
        gnfw_x_lower = (pix_x - center_pix_x - 0.5) * arcseconds_per_pixel
        gnfw_x_upper = (pix_x - center_pix_x + 0.5) * arcseconds_per_pixel
        gnfw_y_lower = (pix_y - center_pix_y - 0.5) * arcseconds_per_pixel
        gnfw_y_upper = (pix_y - center_pix_y + 0.5) * arcseconds_per_pixel
        gnfw_z_lower = lambda x, y: -np.sqrt(inner) if (inner := (5*R500)**2 - x**2 - y**2) > 0 else 0
        gnfw_z_upper = lambda x, y: -gnfw_z_lower(x, y)
        func = lambda z, y, x: f_gnfw_ellipsoid_with_rot(theta, x, y, z, p0, r_x, r_y, r_z, R500)
        pix_val, err = scipy.integrate.tplquad(func, gnfw_x_lower, gnfw_x_upper, gnfw_y_lower, gnfw_y_upper, gnfw_z_lower, gnfw_z_upper, epsabs=epsabs, epsrel=epsrel)
        # get average within pixel because integrated using arcsecond units
        pix_val /= arcseconds_per_pixel**2
        pixels[pix_y, pix_x] = pix_val
    return pixels

# get gnfw(s_xy^2) where s = sqrt(s_xy^2 + (z/r_z)^2))
# s_xy^2 = (x/r_x)^2 + (y/r_y)^2
def interp_gnfw_s_xy_sqr(p0, r_x, r_y, r_z, R500, num_samples=100, epsabs=1.49e-8, epsrel=1.49e-8):
    # because we are integrating only withing sphere of radius 5*R500,
    # max value of s is 5*R500 / min(r_x, r_y, r_z) (can prove with Lagrange multipliers)
    # / max (r_x, r_y, r_z) if we are scaling major axis (we are)
    max_s_xy = 5*R500 / max(r_x, r_y)
    #s_xy_sqr_samples = np.hstack(([0], np.geomspace(0.001, max_s_xy**2, num=num_samples, endpoint=True)))
    s_xy_sqr_samples = np.geomspace(0.000001, max_s_xy**2, num=num_samples, endpoint=True)


    # evaluate gnfw(s_xy^2) at the s_xy samples
    evaluated_gnfw = []
    for s_xy_sqr in s_xy_sqr_samples:
        
        # if we can integrate over ellipsoid (maybe by making either the smallest or largest axis the length of the radius of sphere)
        # s = 5*R500 / min or max (r_x, r_y); min for smallest axis, max for largest axis
        # we will know s, so can solve for z bounds with x, y
        # s^2 = x^2/a^2 + y^2/b^2 + z^2/c^2 => z = sqrt(c^2(s^2 - s_xy^2))

        s = 5*R500 / max(r_x, r_y)
        inner = s**2 - s_xy_sqr
        upper_z_bound = np.abs(r_z) * np.sqrt(inner) if inner > 0 else 0
        evaluated_gnfw.append(
            scipy.integrate.quad(lambda z: f_gnfw_s_xy_sqr_ellipsoid(s_xy_sqr, z, r_z, p0), 0, upper_z_bound, epsabs=epsabs, epsrel=epsrel)[0] * 2
            )
    interp = scipy.interpolate.interp1d(s_xy_sqr_samples, evaluated_gnfw, kind='linear', fill_value=0, bounds_error=False, assume_sorted=True)
    return interp

# ONLY FOR TESTING PURPOSES; TOO SLOW OTHERWISE
def interp_solution(theta, p0, r_x, r_y, r_z, arcseconds_per_pixel, R500, offset_x=0, offset_y=0, img_width=470, img_height=470, num_samples=100, epsabs=1.49e-8, epsrel=1.49e-8):
    # theta is ccw rotation in x,y plane in degrees
    # convert theta to rads
    theta = np.deg2rad(theta)

    center_pix_x = (img_width - 1) / 2 + offset_x
    center_pix_y = (img_height - 1) / 2 + offset_y
    pixels = np.zeros((img_height, img_width))

    gnfw_s_xy_sqr = interp_gnfw_s_xy_sqr(p0, r_x, r_y, r_z, R500, num_samples, epsabs=epsabs, epsrel=epsrel)
    def f_gnfw_s_xy_sqr_with_rot(x, y):
        # x, y are in arcseconds, (0, 0) is center of ellipsoid
        x_rot = x*np.cos(theta) - y*np.sin(theta)
        y_rot = x*np.sin(theta) + y*np.cos(theta)
        s_xy_sqr = (x_rot/r_x)**2 + (y_rot/r_y)**2

        if s_xy_sqr < 0.000001: # 0.000001 = 0.001^2
            s_xy_sqr = 0.000001

        return gnfw_s_xy_sqr(s_xy_sqr)

    for pix_y in range(img_height):
        for pix_x in range(img_width):
            gnfw_x_lower = (pix_x - center_pix_x - 0.5) * arcseconds_per_pixel
            gnfw_x_upper = (pix_x - center_pix_x + 0.5) * arcseconds_per_pixel
            gnfw_y_lower = (pix_y - center_pix_y - 0.5) * arcseconds_per_pixel
            gnfw_y_upper = (pix_y - center_pix_y + 0.5) * arcseconds_per_pixel
            pix_val, err = scipy.integrate.dblquad(f_gnfw_s_xy_sqr_with_rot, gnfw_x_lower, gnfw_x_upper, gnfw_y_lower, gnfw_y_upper, epsabs=epsabs, epsrel=epsrel)
            # get average within pixel because integrated using arcsecond units
            pix_val /= arcseconds_per_pixel**2
            pixels[pix_y, pix_x] = pix_val
        print(pix_y)
    return pixels

# useful
def eval_pixel_centers(theta, p0, r_x, r_y, r_z, arcseconds_per_pixel, R500, offset_x=0, offset_y=0, img_height=470, img_width=470, num_samples=100, epsabs=1.49e-8, epsrel=1.49e-8):
    
    gnfw_s_xy_sqr = interp_gnfw_s_xy_sqr(p0, r_x, r_y, r_z, R500, num_samples, epsabs=epsabs, epsrel=epsrel)
    
    # offset in arcseconds
    # theta is ccw rotation in x,y plane in degrees
    # convert theta to rads
    return eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, arcseconds_per_pixel, offset_x, offset_y, img_height, img_width)

# evaluating the interp takes a long time but is reusable
# if we want to evaluate the same model at different resolutions, we can first find the interp
# then, call this function
def eval_pixel_centers_use_interp(gnfw_s_xy_sqr, theta, r_x, r_y, arcseconds_per_pixel, offset_x, offset_y, img_height, img_width):
    # offset in arcseconds
    # theta is ccw rotation in x,y plane in degrees
    # convert theta to rads
    theta = np.deg2rad(theta)

    center_pix_x = (img_width - 1) / 2 + (offset_x / arcseconds_per_pixel)
    center_pix_y = (img_height - 1) / 2 + (offset_y / arcseconds_per_pixel)
    pixels = np.zeros((img_height, img_width))
    def f_gnfw_s_xy_sqr_with_rot(x, y):
        # x, y are in arcseconds, (0, 0) is center of ellipsoid
        x_rot = x*np.cos(theta) - y*np.sin(theta)
        y_rot = x*np.sin(theta) + y*np.cos(theta)
        s_xy_sqr = (x_rot/r_x)**2 + (y_rot/r_y)**2

        if s_xy_sqr < 0.000001: # 0.000001 = 0.001^2
            s_xy_sqr = 0.000001

        return gnfw_s_xy_sqr(s_xy_sqr)

    for pix_y in range(img_height):
        for pix_x in range(img_width):
            gnfw_x = (pix_x - center_pix_x) * arcseconds_per_pixel
            gnfw_y = (pix_y - center_pix_y) * arcseconds_per_pixel
            pix_val = f_gnfw_s_xy_sqr_with_rot(gnfw_x, gnfw_y)
            pixels[pix_y, pix_x] = pix_val
    return pixels



# https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array
def rebin_2d(arr, bin_shape):
        a, b = bin_shape
        return arr.reshape(arr.shape[0] // a, a, arr.shape[1] // b, b).mean(axis=(1,3))

if __name__ == "__main__":
    from conversions import convert_microkelvin_to_mjysr
    import di_utils
    from matplotlib import pyplot as plt

    theta = 0

    R500 = 200
    R2500 = 66.7104
    params = [-2.80218707e+00, -4.67692923e+00,  4.04650453e+02, -2.67669039e-02,
 -3.85902294e-03,  3.94686241e+01,  2.64345529e+01]
    errors = [  1.74265301,   3.04220972, 106.55782932,   0.31058793,   0.31098361,
   4.10405139,   5.36751085]
    P0_150, P0_90, RS, _, _, c_150, c_90 = params
    err_150, err_90, _, _, _, _, _ = errors

    # try with 1; should be same as sphere
    to_eval = [(x, 15) for x in range(15, 30)]
    naive_gnfw_fits_150 = full_solution(to_eval, theta, P0_150, RS, RS, RS, 4, R500, 0, 0, 30, 30)
    plt.figure(-1)
    plt.title('naive')
    plt.imshow(naive_gnfw_fits_150)
    #cProfile.run('interp_solution(theta, P0_150, RS, RS, RS, 4, R500, 0, 0, 470, 470, 100, 1e-1, 1e-1)', 'cProfile', 'cumulative')
    #gnfw_fits_150 = interp_solution(theta, P0_150, RS, RS, RS, 4, R500, 0, 0, 30, 30, 100)
    #np.save('gnfw_fits_150_ellipsoidal_interped', gnfw_fits_150)
    gnfw_fits_150 = np.load('gnfw_fits_150_ellipsoidal_interped.npy')
    
    gnfw_fits_150_no_dbl = eval_pixel_centers(theta, P0_150, RS, RS, RS, 4, R500, 0, 0, 30, 30, 100)
    
    gnfw_fits_150_no_dbl_1_arc = eval_pixel_centers(theta, P0_150, RS, RS, RS, 4/4, R500, 0, 0, 30*4, 30*4, 100)
    
    # rebin into 4 arcseconds
    rebinned_1_arc = rebin_2d(gnfw_fits_150_no_dbl_1_arc, (4, 4))

    gnfw_fits_150_no_dbl_2_arc = eval_pixel_centers(theta, P0_150, RS, RS, RS, 4/2, R500, 0, 0, 30*2, 30*2, 100)
    rebinned_2_arc = rebin_2d(gnfw_fits_150_no_dbl_2_arc, (2, 2))


    import eval_gnfw
    model_150_no_c = eval_gnfw.make_los_gnfw_grid(P0_150, RS, R500,
                        30, 30,
                        0, 0, arcseconds_per_pixel=4, epsrel=1e-2,
                        num_processes=4)
    plt.figure(0)
    plt.title('ellipsoid dbl-integrated')
    plt.imshow(gnfw_fits_150)
    plt.figure(1)
    plt.title('sphere')
    plt.imshow(model_150_no_c)
    plt.figure(10)
    plt.title('rebinned 1 arcsecond')
    plt.imshow(rebinned_1_arc)
    plt.figure(11)
    plt.title('rebinned 2 arcsecond')
    plt.imshow(rebinned_2_arc)
    plt.figure(2)
    plt.title('difference dbl-ellipsoidal/spherical')
    plt.imshow(gnfw_fits_150/model_150_no_c)
    plt.figure(3)
    plt.title('ellipsoidal no-dbl')
    plt.imshow(gnfw_fits_150_no_dbl)
    plt.figure(4)
    plt.title('difference dbl-ellipsoidal/no-dbl-ellipsoidal')
    plt.imshow(gnfw_fits_150/gnfw_fits_150_no_dbl)
    plt.figure(20)
    plt.title('difference dbl-ellipsoidal/rebinned 2 arc')
    plt.imshow(gnfw_fits_150/rebinned_2_arc)
    plt.figure(21)
    plt.title('difference dbl-ellipsoidal/rebinned 1 arc')
    plt.imshow(gnfw_fits_150/rebinned_1_arc)
    plt.show()

    # tests show that for the 470 * 470 grid at 4 arcseconds/pixel, evaluating the center of each pixel is sufficient (within 1%)







    #cProfile.run('interp_solution(theta, P0_150, RS, RS, RS, 30, R500, 0, 0, 30, 30, 100, 1e-2, 1e-2)', 'cProfile', 'cumulative')
    #gnfw_fits_150 = interp_solution(theta, P0_150, RS, RS, RS, 30, R500, 0, 0, 30, 30, 100, 1e-2, 1e-2)
    di_150 = di_utils.get_R2500_avg(gnfw_fits_150, 4, R2500)
    normalized = gnfw_fits_150 / di_150
    print('di 150', di_150)
    print('avg within R2500 for normalized', di_utils.get_R2500_avg(normalized, 4, R2500))

    # and, we can get an approximation of the sigma di
    sigma_150 = di_utils.calc_sigma_di(150, err_150, P0_150, di_150)
    print('sigma di 150', sigma_150)
    
    plt.figure(0)
    plt.title('fits data')
    plt.imshow(normalized)
    plt.show()
    #print(np.sum(np.abs(np.load('normalized_gnfw_model.npy') - normalized)))
    
    

