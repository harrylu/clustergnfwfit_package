import emcee
import numpy as np
import matplotlib.pyplot as plt

filename = 'emcee_incorrect_theta.h5'
reader = emcee.backends.HDFBackend(filename)

# tau = reader.get_autocorr_time()
# burnin = int(2 * np.max(tau))
# thin = int(0.5 * np.min(tau))
samples = reader.get_chain()
# log_prob_samples = reader.get_log_prob(flat=True)
# log_prior_samples = reader.get_blobs(flat=True)
pnames = ['theta', 'p0_90', 'p0_150', 'r_x', 'r_y', 'offset_x', 'offset_y', 'c_90', 'c_150']
p_lower_bounds = [90, -5000, -5000, 10, 10, -100, -100, None, None]
p_upper_bounds = [270, 5000, 5000, 1000, 1000, 100, 100, None, None]
subplots = [list(plt.subplots(3, 1)) for _ in range(len(pnames))]
for i, (fig, *_) in enumerate(subplots):
    fig.suptitle(pnames[i])
    for ax in fig.axes:
        ax.set_ybound(p_lower_bounds[i], p_upper_bounds[i])

num_samples = len(samples)
# samples is shape (nsamples, nwalkers, ndims)
for i in range(samples.shape[2]):
    _, (ax1, ax2, ax3) = subplots[i]
    pdata = samples[:, :, i]
    # pdata is shape (nsamples, nwalkers)
    for walker_idx in range(pdata.shape[1]):
        y = pdata[:, walker_idx]
        ax1.plot(y, alpha=0.1)
        ax2.plot(y, color='r', alpha=0.02)
    ax3.plot(np.mean(pdata, axis=1))
        
        
plt.show()


print("flat chain shape: {0}".format(samples.shape))
# print("flat log prob shape: {0}".format(log_prob_samples.shape))
# print("flat log prior shape: {0}".format(log_prior_samples.shape))