import mcmc_post

if __name__ == "__main__":
    # pnames = ['theta', 'cbrt_p0_90', 'cbrt_p0_150', 'r_x', 'r_y', 'offset_x', 'offset_y', 'c_90', 'c_150', 'cbrt_p0_bolocam', 'c_bolocam']
    # make_trace(pnames, '/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4', 'backend.h5', 'lower_bounds.txt', 'upper_bounds.txt')
    # show_pickled_plots('/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4/mcmc_trace')
    
    # pnames = ['theta', 'cbrt_p0_90', 'cbrt_p0_150', 'r_x', 'r_y', 'offset_x', 'offset_y', 'c_90', 'c_150', 'cbrt_p0_bolocam', 'c_bolocam']
    # labels = list(pnames)
    # labels += ["log prob", "log prior"]
    # make_corner(labels, '/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4_act_covar_20k_3_iqr_faster_interp', 'backend.h5')
    # show_pickled_plots('/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4_act_covar_20k_3_iqr_faster_interp/corner_plots')

    # mcmc_post.show_pickled_plots('/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4/corner_plots')

    mcmc_post.show_npy_plots('/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4_act_covar_20k_3_iqr_faster_interp/maps_and_covariance')
    mcmc_post.show_pickled_plots('/home/harry/clustergnfwfit_package/run_outputs/MACSJ0025.4_act_covar_20k_3_iqr_faster_interp/corner_plots')