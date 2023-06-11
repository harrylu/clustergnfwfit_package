import emcee
import numpy as np
import matplotlib.pyplot as plt

filename = 'emcee_backend_cube_root.h5'
reader = emcee.backends.HDFBackend(filename)

# tau = reader.get_autocorr_time()
# burnin = int(2 * np.max(tau))
# thin = int(0.5 * np.min(tau))
samples = reader.get_chain()
log_prob_samples = reader.get_log_prob()
log_prior_samples = reader.get_blobs()

pnames = ['theta', 'p0_90', 'p0_150', 'r_x', 'r_y', 'offset_x', 'offset_y', 'c_90', 'c_150']
p_lower_bounds = [90, -5000, -5000, 10, 10, -100, -100, None, None]
p_upper_bounds = [270, 5000, 5000, 1000, 1000, 100, 100, None, None]
subplots = [list(plt.subplots(3, 1)) for _ in range(len(pnames))]
for i, (fig, *_) in enumerate(subplots):
    fig.suptitle(pnames[i])
    for ax in fig.axes:
        lower = p_lower_bounds[i]
        upper = p_upper_bounds[i]
        if lower is not None or upper is not None:
            ax.set_ylim(bottom=lower, top=upper)

num_samples = samples.shape[0]
nwalkers = samples.shape[1]
ndims = samples.shape[2]
# samples is shape (nsamples, nwalkers, ndims)

# shape is (nsamples, nwalkers)
chisq_samples = log_prob_samples * -2
# chisq_min = np.expand_dims(np.min(chisq_samples, axis=1), 1)
# chisq_max = np.expand_dims(np.max(chisq_samples, axis=1), 1)
# chisq_range = chisq_max - chisq_min
# alphas = 0.005*np.divide(chisq_samples - np.tile(chisq_min, (1, nwalkers)), np.tile((chisq_range), (1, nwalkers)))

# i is param #
for i in range(ndims):
    _, (ax1, ax2, ax3) = subplots[i]
    pdata = samples[:, :, i]
    # pdata is shape (nsamples, nwalkers)
    for walker_idx in range(pdata.shape[1]):
        y = pdata[:, walker_idx]
        ax1.plot(y, alpha=0.1)
        ax2.plot(y, color='r', alpha=0.02)
        # ax2.scatter(range(num_samples), y, color='r', alpha=alphas[:, i])
    ax3.plot(np.mean(pdata, axis=1))
    ax3.set_ylim(ax1.get_ylim())

# walker chi-sqs
fig_chisq, (ax_chisq_walkers, ax_chisq_mean) = plt.subplots(2, 1)
ax_chisq_walkers.plot(chisq_samples, alpha=0.1)
ax_chisq_mean.plot(np.mean(chisq_samples, axis=1))



print("chain shape: {0}".format(samples.shape))
print("log prob shape: {0}".format(log_prob_samples.shape))
print("log prior shape: {0}".format(log_prior_samples.shape))   
plt.show()

