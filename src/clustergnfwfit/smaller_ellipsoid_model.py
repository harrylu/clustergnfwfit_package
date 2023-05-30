# tests ellipsoid model with 30 arcsecond maps

from matplotlib import pyplot as plt
import time


from clustergnfwfit.ellipsoid_model import *
if __name__ == "__main__":
    theta = 0

    R500 = 200
    R2500 = 66.7104
    params = [-2.80218707e+00, -4.67692923e+00,  4.04650453e+02, -2.67669039e-02,
 -3.85902294e-03,  3.94686241e+01,  2.64345529e+01]
    errors = [  1.74265301,   3.04220972, 106.55782932,   0.31058793,   0.31098361,
   4.10405139,   5.36751085]
    P0_150, P0_90, RS, _, _, c_150, c_90 = params
    err_150, err_90, _, _, _, _, _ = errors

    r_x = RS
    r_y = RS*0.5
    r_z = np.sqrt(r_x*r_y)

    # eval full solution @ 5 pixel line from center
    # 20 x 20 map of 30 arcsecond pixels
    timer = time.process_time()
    model_1_arc = eval_pixel_centers(theta, P0_150, r_x, r_y, r_z, 30/30, R500, 0, 0, 20*30, 20*30)
    rebinned_1 = rebin_2d(model_1_arc, (30, 30))
    print(f'1": {time.process_time() - timer}')

    timer = time.process_time()
    model_2_arc = eval_pixel_centers(theta, P0_150, r_x, r_y, r_z, 30/15, R500, 0, 0, 20*15, 20*15)
    rebinned_2 = rebin_2d(model_2_arc, (15, 15))
    print(f'2": {time.process_time() - timer}')

    timer = time.process_time()
    model_5_arc = eval_pixel_centers(theta, P0_150, r_x, r_y, r_z, 30/6, R500, 0, 0, 20*6, 20*6)
    rebinned_5 = rebin_2d(model_5_arc, (6, 6))
    print(f'5": {time.process_time() - timer}')

    timer = time.process_time()
    model_10_arc = eval_pixel_centers(theta, P0_150, r_x, r_y, r_z, 30/3, R500, 0, 0, 20*3, 20*3)
    rebinned_10 = rebin_2d(model_10_arc, (3, 3))
    print(f'10": {time.process_time() - timer}')
    
    timer = time.process_time()
    model_30_arc = eval_pixel_centers(theta, P0_150, r_x, r_y, r_z, 30/1, R500, 0, 0, 20*1, 20*1)
    print(f'30": {time.process_time() - timer}')

    timer = time.process_time()
    to_eval = [(x, 10) for x in range(10, 20)]
    truth = full_solution(to_eval, theta, P0_150, r_x, r_y, r_z, 30, R500, 0, 0, 20, 20)

    plt.figure(0)
    plt.title('truth')
    plt.imshow(truth)
    plt.figure(1)
    plt.title('1 arcseconds binned into 30')
    plt.imshow(rebinned_1)
    plt.figure(2)
    plt.title('2 arcseconds binned into 30')
    plt.imshow(rebinned_2)
    plt.figure(5)
    plt.title('5 arcseconds binned into 30')
    plt.imshow(rebinned_5)
    plt.figure(10)
    plt.title('10 arcseconds binned into 30')
    plt.imshow(rebinned_10)
    plt.figure(30)
    plt.title('30 arcseconds no rebin')
    plt.imshow(model_30_arc)

    plt.figure(101)
    plt.title('1 arc / truth')
    plt.imshow(rebinned_1 / truth)
    plt.figure(102)
    plt.title('2 arc / truth')
    plt.imshow(rebinned_2 / truth)
    plt.figure(105)
    plt.title('5 arc / truth')
    plt.imshow(rebinned_5 / truth)
    plt.figure(110)
    plt.title('10 arc / truth')
    plt.imshow(rebinned_10 / truth)
    plt.title('30 arc / truth')
    plt.imshow(model_30_arc / truth)
    plt.show()
    