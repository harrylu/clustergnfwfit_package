import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    save_path = 'data/covar_np_saves'

    covar_90 = np.load(os.path.join(save_path, 'act_90_covar_2000.npy'))
    covar_150 = np.load(os.path.join(save_path, 'act_150_covar_2000.npy'))

    covar_bolocam = np.load(os.path.join(save_path, 'bolo_covar_1000.npy'))

    eig_vals, eig_vecs = np.linalg.eigh(covar_bolocam)
    eig_vals[eig_vals < np.max(eig_vals) * 1e-7] = np.inf
    # eig_vecs should probably be orthonormal, so can probably .T instead of .inv
    # covar_bolocam_reformed = eig_vecs @ np.diag(eig_vals) @ np.linalg.inv(eig_vecs)

    eig_vals_inv = 1/eig_vals
    covar_bolocam_inv = eig_vecs.T @ np.diag(eig_vals_inv) @ eig_vecs


    plt.figure('before reform')
    plt.imshow(covar_bolocam)
    # plt.figure('after reform')
    # plt.imshow(covar_bolocam_reformed)
    # plt.figure('diff')
    # plt.imshow(covar_bolocam - covar_bolocam_reformed)
    # print(f'diff sum: {np.sum(covar_bolocam - covar_bolocam_reformed)}')

    print(np.linalg.inv(covar_bolocam))
    # print(np.linalg.inv(covar_bolocam_reformed))

    plt.figure('bolocam covar inv')
    plt.imshow(covar_bolocam_inv)


    plt.figure('eig vals bolocam')
    plt.imshow(np.real(np.linalg.eigvals(covar_bolocam)).reshape((42, 42)))

    act_90_covar_median = np.load(os.path.join(save_path, 'act_90_covar_2000_median.npy'))
    plt.figure('90 median')
    plt.imshow(act_90_covar_median)

    act_150_covar_median = np.load(os.path.join(save_path, 'act_150_covar_2000_median.npy'))
    plt.figure('150 median')
    plt.imshow(act_150_covar_median)

    plt.figure('covar 90')
    plt.imshow(covar_90)
    plt.figure('covar 150')
    plt.imshow(covar_150)

    plt.figure('eig vals 90')
    n = int(np.sqrt(covar_90.shape[0]))
    plt.imshow(np.real(np.linalg.eigvals(covar_90)).reshape((n, n)))

    plt.figure('150 inv')
    plt.imshow(np.linalg.inv(covar_150))

    plt.show()
